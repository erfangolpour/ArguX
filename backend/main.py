# backend/main.py
import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from random import randint
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse
from uuid import uuid4

import aiohttp
import cv2
import numpy as np
import redis
import redis.asyncio as aioredis  # Use async redis client
from bs4 import BeautifulSoup
from celery import Celery, states
from celery.exceptions import CeleryError
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl
from redis.exceptions import RedisError

# Import settings
from settings import settings
from ultralytics import YOLO
from ultralytics import settings as yolo_ultralytics_settings  # Avoid name clash

# --- Logging Configuration ---
logging.basicConfig(level=settings.log_level.upper(), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Global Clients and State ---
celery_app = Celery("camera_scan", broker=settings.celery_broker_url, backend=settings.celery_result_backend)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    broker_connection_retry_on_startup=True,
    # Optional: Task routing, rate limits, etc. can be configured here
    # task_routes = {'backend.main.process_camera_task': {'queue': 'image_processing'}},
)
# Optional: Auto-discover tasks (if you move tasks to separate files later)
# celery_app.autodiscover_tasks(['backend.tasks'])
redis_client: aioredis.Redis | None = None
http_client_session: aiohttp.ClientSession | None = None
model_cache: Dict[str, YOLO] = {}
connected_clients: Set[WebSocket] = set()
valid_models: List[str] = []
app_state: Dict[str, Any] = {"model_name": settings.default_model}  # Store global app state like selected model


# --- Pydantic Models ---
class GeoLocation(BaseModel):
    country: Optional[str] = "Unknown"
    countryCode: Optional[str] = "XX"
    regionName: Optional[str] = "Unknown"
    city: Optional[str] = "Unknown"
    lat: Optional[float] = None
    lon: Optional[float] = None
    query: Optional[str] = "Unknown"  # IP Address
    org: Optional[str] = "Unknown"
    as_info: Optional[str] = Field(None, alias="as")  # Use alias for 'as'


class Detection(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class CameraScanResult(BaseModel):
    id: str  # Celery Task ID
    url: HttpUrl
    geo: Optional[GeoLocation] = None
    detections: List[Detection] = []
    snapshot: Optional[str] = None  # Base64 encoded image string
    timestamp: datetime = Field(default_factory=datetime.now)
    processed: bool = False  # Indicates if processing was successful
    error: Optional[str] = None  # Store processing error message


class SuccessResponse(BaseModel):
    status: str = "success"
    message: Optional[str] = None
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
    detail: Optional[str] = None


class ScanStatusResponse(BaseModel):
    scan_running: bool
    active_tasks: int
    selected_model: str


# --- Lifespan Management (Startup/Shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, http_client_session, valid_models
    logger.info("Application startup...")

    # Initialize Redis client
    try:
        redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await redis_client.ping()
        logger.info("Redis client connected.")
    except RedisError as e:
        logger.critical(f"Failed to connect to Redis: {e}", exc_info=True)
        # Depending on requirements, you might want to exit here or run in a degraded state
        redis_client = None  # Ensure client is None if connection failed

    # Initialize HTTP client session
    http_client_session = aiohttp.ClientSession(
        headers={"User-Agent": settings.http_user_agent},
        timeout=aiohttp.ClientTimeout(total=settings.http_client_timeout),
        connector=aiohttp.TCPConnector(limit=100),  # Increase connection pool size
    )
    logger.info("Aiohttp client session created.")

    # Configure Ultralytics settings and find valid models
    yolo_ultralytics_settings.update(
        {
            "weights_dir": settings.yolo_weights_dir,
            "runs_dir": settings.yolo_runs_dir,
            "uuid": str(uuid4()),  # Prevent potential conflicts with multiple instances
            "sync": False,  # Disable sync for better performance in server env
        }
    )
    valid_models = [f.name for f in settings.yolo_base_dir.iterdir() if f.is_file() and f.suffix == ".pt"]
    logger.info(f"Valid YOLO models found: {valid_models}")
    if settings.default_model not in valid_models and valid_models:
        logger.warning(f"Default model '{settings.default_model}' not found. Using '{valid_models[0]}'.")
        settings.default_model = valid_models[0]
        app_state["model_name"] = settings.default_model
    elif not valid_models:
        logger.error("No valid YOLO models found in YOLO directory!")
        # Potentially raise an error or prevent startup

    logger.info("Application startup complete.")
    yield  # Application runs here
    logger.info("Application shutdown...")

    # Clean up resources
    if http_client_session:
        await http_client_session.close()
        logger.info("Aiohttp client session closed.")
    if redis_client:
        await redis_client.close()
        logger.info("Redis client closed.")
    # Celery does not require explicit shutdown here

    logger.info("Application shutdown complete.")


# --- FastAPI App Instance ---
app = FastAPI(title=settings.app_name, description=settings.app_description, lifespan=lifespan)

# --- Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# --- Dependency Injection ---
async def get_redis_client() -> aioredis.Redis:
    if redis_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis connection not available")
    return redis_client


async def get_http_session() -> aiohttp.ClientSession:
    if http_client_session is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="HTTP client session not available")
    return http_client_session


def get_celery() -> Celery:
    if celery_app is None:
        # This case should ideally not happen anymore if defined globally
        logger.critical("Celery app is unexpectedly None during dependency injection!")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Celery service not configured correctly."
        )
    return celery_app


# --- Custom Exception Handler ---
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(message="Internal Server Error", detail=str(exc)).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(message="Request Error", detail=exc.detail).model_dump(),
        headers=exc.headers,
    )


@app.exception_handler(RedisError)
async def redis_exception_handler(request, exc: RedisError):
    logger.error(f"Redis error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            message="Service Unavailable", detail="Could not connect to backend data store."
        ).model_dump(),
    )


@app.exception_handler(aiohttp.ClientError)
async def aiohttp_exception_handler(request, exc: aiohttp.ClientError):
    logger.error(f"HTTP client error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            message="Service Unavailable", detail=f"Could not connect to external service: {exc}"
        ).model_dump(),
    )


# --- Redis Task Management Helpers ---
async def add_task_to_redis(task_id: str, redis: aioredis.Redis):
    try:
        await redis.sadd(settings.active_tasks_key, task_id)
        logger.debug(f"Added task {task_id} to Redis set {settings.active_tasks_key}")
    except RedisError as e:
        logger.error(f"Failed to add task {task_id} to Redis: {e}", exc_info=True)
        # Decide how to handle this - potentially raise an exception or just log


async def remove_task_from_redis(task_id: str, redis: aioredis.Redis):
    try:
        await redis.srem(settings.active_tasks_key, task_id)
        logger.debug(f"Removed task {task_id} from Redis set {settings.active_tasks_key}")
    except RedisError as e:
        logger.error(f"Failed to remove task {task_id} from Redis: {e}", exc_info=True)


async def get_active_tasks_from_redis(redis: aioredis.Redis) -> Set[str]:
    try:
        tasks = await redis.smembers(settings.active_tasks_key)
        return tasks
    except RedisError as e:
        logger.error(f"Failed to get active tasks from Redis: {e}", exc_info=True)
        return set()


async def get_active_tasks_count_from_redis(redis: aioredis.Redis) -> int:
    try:
        return await redis.scard(settings.active_tasks_key)
    except RedisError as e:
        logger.error(f"Failed to get active task count from Redis: {e}", exc_info=True)
        return 0  # Return 0 if count cannot be retrieved


async def is_scan_running(redis: aioredis.Redis) -> bool:
    try:
        return await redis.exists(settings.scan_running_key) > 0
    except RedisError as e:
        logger.error(f"Failed to check scan status in Redis: {e}", exc_info=True)
        return False  # Assume not running if check fails


# --- YOLO Model Loader ---
def get_yolo_model(model_name: str) -> YOLO:
    if model_name not in model_cache:
        model_path = settings.yolo_base_dir / model_name
        if not model_path.exists():
            # This should ideally be caught earlier, but double-check
            logger.error(f"YOLO model file not found: {model_path}")
            raise ValueError(f"Model file '{model_name}' not found.")
        try:
            logger.info(f"Loading YOLO model: {model_name}")
            model_cache[model_name] = YOLO(str(model_path))
            logger.info(f"YOLO model loaded successfully: {model_name}")
        except Exception as e:
            logger.critical(f"Failed to load YOLO model {model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Could not load model '{model_name}'.") from e
    return model_cache[model_name]


# --- Camera Processing Logic ---
async def process_camera_stream(session: aiohttp.ClientSession, cam_url: str, model_name: str) -> CameraScanResult:
    """
    Fetches image, geo-data, performs detection, and returns structured result.
    Designed to be run within an async context (like the Celery task runner).
    """
    task_id = uuid4().hex  # Placeholder ID if not run via Celery context
    result = CameraScanResult(id=task_id, url=cam_url)  # Initialize result object

    try:
        # 1. Fetch Geolocation Data (Best effort)
        try:
            ip_addr = urlparse(cam_url).netloc.split(":")[0]
            # Use an async request within the provided session
            async with session.get(f"http://ip-api.com/json/{ip_addr}") as geo_resp:
                if geo_resp.ok:
                    geo_data = await geo_resp.json()
                    # Validate and assign geo data using Pydantic model
                    result.geo = GeoLocation.model_validate(geo_data)
                    logger.debug(f"Geo data fetched for {ip_addr}: {result.geo}")
                else:
                    logger.warning(f"Failed to fetch geo data for {ip_addr}: Status {geo_resp.status}")
                    result.geo = GeoLocation()  # Assign default unknown geo data
        except aiohttp.ClientError as e:
            logger.warning(f"Network error fetching geo data for {ip_addr}: {e}")
            result.geo = GeoLocation()
        except Exception as e:
            logger.error(f"Unexpected error fetching geo data for {ip_addr}: {e}", exc_info=True)
            result.geo = GeoLocation()

        # 2. Fetch and Decode Image Frame
        frame = None
        try:
            async with session.get(cam_url) as resp:
                if not resp.ok:
                    error_msg = f"Failed to fetch camera image: Status {resp.status}"
                    logger.warning(f"{error_msg} for URL: {cam_url}")
                    result.error = error_msg
                    return result  # Return early, cannot process

                if resp.content_type not in ("image/jpeg", "image/png", "image/gif"):
                    logger.warning(f"Unexpected content type '{resp.content_type}' for URL: {cam_url}")
                    # Try to decode anyway, might work for some streams
                    # If it's crucial to only process specific types, return an error here

                frame_data = await resp.read()
                frame_np = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

                if frame is None:
                    error_msg = "Failed to decode image frame."
                    logger.warning(f"{error_msg} for URL: {cam_url}")
                    result.error = error_msg
                    # Include snapshot of raw data if helpful for debugging? Potentially large.
                    # result.snapshot = f"data:{resp.content_type};base64,{base64.b64encode(frame_data).decode('utf-8')}"
                    return result  # Return early

        except asyncio.TimeoutError:
            error_msg = "Timeout fetching camera image."
            logger.warning(f"{error_msg} for URL: {cam_url}")
            result.error = error_msg
            return result
        except aiohttp.ClientError as e:
            error_msg = f"Network error fetching camera image: {e}"
            logger.warning(f"{error_msg} for URL: {cam_url}")
            result.error = error_msg
            return result
        except cv2.error as e:
            error_msg = f"OpenCV error decoding image: {e}"
            logger.warning(f"{error_msg} for URL: {cam_url}")
            result.error = error_msg
            return result
        except Exception as e:
            error_msg = f"Unexpected error fetching/decoding image: {e}"
            logger.error(f"{error_msg} for URL: {cam_url}", exc_info=True)
            result.error = error_msg
            return result

        # 3. Perform Object Detection
        try:
            yolo_model = get_yolo_model(model_name)
            # Run prediction (ensure frame is valid)
            predictions = yolo_model.predict(frame, conf=0.5, verbose=False)  # verbose=False reduces console spam

            processed_frame = frame.copy()  # Work on a copy for plotting

            for yolo_result in predictions:
                # Plot bounding boxes on the *copy* of the frame
                processed_frame = yolo_result.plot(img=processed_frame)

                # Extract detection details
                for box in yolo_result.boxes:
                    try:
                        label = yolo_result.names[int(box.cls[0])]
                        confidence = round(float(box.conf[0]), 2)
                        result.detections.append(Detection(label=label, confidence=confidence))
                    except (IndexError, ValueError, KeyError) as e:
                        logger.warning(
                            f"Error parsing detection box for {cam_url}: {e} - Box: {box.data}", exc_info=True
                        )
                        continue  # Skip this box if parsing fails

            # 4. Encode Snapshot with BBoxes
            is_success, buffer = cv2.imencode(".jpg", processed_frame)
            if is_success:
                result.snapshot = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
            else:
                logger.warning(f"Failed to encode processed frame to JPEG for {cam_url}")
                # Optionally encode the original frame as fallback?
                is_success_orig, buffer_orig = cv2.imencode(".jpg", frame)
                if is_success_orig:
                    result.snapshot = f"data:image/jpeg;base64,{base64.b64encode(buffer_orig).decode('utf-8')}"

            result.processed = True  # Mark as successfully processed
            logger.info(f"Successfully processed {cam_url}. Detections: {len(result.detections)}")

        except (RuntimeError, ValueError) as e:  # Catch model loading or prediction errors
            error_msg = f"YOLO processing error: {e}"
            logger.error(f"{error_msg} for URL: {cam_url}", exc_info=True)
            result.error = error_msg
        except cv2.error as e:
            error_msg = f"OpenCV error during plotting/encoding: {e}"
            logger.error(f"{error_msg} for URL: {cam_url}", exc_info=True)
            result.error = error_msg
        except Exception as e:
            error_msg = f"Unexpected error during detection/encoding: {e}"
            logger.error(f"{error_msg} for URL: {cam_url}", exc_info=True)
            result.error = error_msg

    except Exception as e:
        # Catch-all for unexpected errors in the main flow
        error_msg = f"Unexpected top-level error processing {cam_url}: {e}"
        logger.critical(error_msg, exc_info=True)
        result.error = error_msg

    # Always set timestamp before returning
    result.timestamp = datetime.now()
    return result


# --- Celery Task ---
# Note: Celery tasks run synchronously within the worker process.
# We use asyncio.run to execute our async processing function.
@celery_app.task(bind=True, name="process_camera_task", ignore_result=False)
def process_camera_task(self, cam_url: str, model_name: str) -> dict:
    """Celery task to process a single camera URL."""
    task_id = self.request.id
    logger.info(f"Task {task_id}: Starting processing for {cam_url} with model {model_name}")

    # Check global scan state (using synchronous redis client for Celery task)
    # This is a simplification; ideally Celery worker would use async redis too,
    # but standard redis client is often used in Celery tasks for simplicity.
    sync_redis_client = redis.Redis(
        host=settings.redis_host, port=settings.redis_port, db=settings.redis_db, decode_responses=True
    )
    try:
        if not sync_redis_client.exists(settings.scan_running_key):
            logger.warning(f"Task {task_id}: Scan stopped globally. Revoking task for {cam_url}.")
            # It's better to let the task finish quickly than try to revoke mid-flight typically.
            # Set a state that indicates it was cancelled due to scan stop.
            self.update_state(state="SCAN_STOPPED", meta={"url": cam_url})
            # Explicitly remove from active set as it won't complete successfully
            sync_redis_client.srem(settings.active_tasks_key, task_id)
            return {"id": task_id, "url": cam_url, "error": "Scan stopped by user", "processed": False}
    except RedisError as e:
        logger.error(f"Task {task_id}: Redis error checking scan status: {e}. Proceeding with task.", exc_info=True)
        # Proceed, hoping the scan is still running or the error is temporary

    # We need an async context to run process_camera_stream
    # Create a temporary session for this task execution
    async def run_processing():
        # Pass task_id down to the processing function
        async with aiohttp.ClientSession(
            headers={
                "User-Agent": settings.http_user_agent,
                "Accept": "image/jpeg,image/png,image/*",
                "Connection": "close",
            },
            timeout=aiohttp.ClientTimeout(total=settings.http_client_timeout),
            connector=aiohttp.TCPConnector(limit=1),
        ) as session:
            result = await process_camera_stream(session, cam_url, model_name)
            result.id = task_id  # Ensure the result has the correct task ID
            return result.model_dump(mode="json")  # Return Pydantic model as dict

    try:
        result_dict = asyncio.run(run_processing())
        if result_dict.get("error"):
            logger.warning(f"Task {task_id}: Processing failed for {cam_url}. Error: {result_dict['error']}")
            # Let Celery mark as SUCCESS but include error details in the result
            # This allows the frontend to know *why* it failed.
        else:
            logger.info(f"Task {task_id}: Processing successful for {cam_url}")

        return result_dict  # Return the result dictionary

    except Exception as e:
        logger.critical(
            f"Task {task_id}: Unexpected failure running async processing for {cam_url}: {e}", exc_info=True
        )
        # Mark task as failed
        self.update_state(
            state=states.FAILURE,
            meta={
                "exc_type": type(e).__name__,
                "exc_message": str(e),
                "url": cam_url,
            },
        )
        # Explicitly remove from active set on failure
        sync_redis_client.srem(settings.active_tasks_key, task_id)
        # Return an error structure consistent with CameraScanResult
        return {
            "id": task_id,
            "url": cam_url,
            "processed": False,
            "error": f"Task execution failed: {type(e).__name__}: {str(e)}",
            "timestamp": datetime.now().isoformat(),  # Ensure timestamp is serializable
        }
    finally:
        # Close the synchronous Redis client connection for this task
        sync_redis_client.close()


# --- Camera Discovery ---
async def fetch_camera_urls_from_source(
    session: aiohttp.ClientSession, country: Optional[str] = None, page: int = 1
) -> List[str]:
    """Fetches a list of camera URLs from Insecam for a given country and page."""
    tag_filter = "bynew" if country is None else f"bycountry/{country}"
    api_endpoint = f"http://www.insecam.org/en/{tag_filter}/?page={page}"
    urls = []
    try:
        logger.debug(f"Fetching camera URLs from: {api_endpoint}")
        async with session.get(api_endpoint) as resp:
            if resp.ok:
                soup = BeautifulSoup(await resp.text(), "html.parser")
                # Correctly find image URLs, handle potential errors
                img_tags = soup.select("img.thumbnail-item__img[src]")  # More specific selector
                urls = [img["src"] for img in img_tags if img["src"].startswith("http")]
                logger.debug(f"Found {len(urls)} URLs on page {page} for filter '{tag_filter}'.")
            else:
                logger.warning(f"Failed to fetch camera list from {api_endpoint}: Status {resp.status}")
                if resp.status == 404 and country:
                    # Specific handling for invalid country code
                    raise ValueError(f"Country code '{country}' not found on Insecam.")
    except aiohttp.ClientError as e:
        logger.error(f"Network error fetching camera list: {e}", exc_info=True)
        # Depending on policy, might want to retry or raise
    except Exception as e:
        logger.error(f"Error parsing camera list page {api_endpoint}: {e}", exc_info=True)
    return urls


async def camera_discovery_loop(
    country: Optional[str], model_name: str, redis: aioredis.Redis, session: aiohttp.ClientSession, celery: Celery
):
    """The main loop for discovering cameras and queueing tasks."""
    logger.info(f"Starting camera discovery loop. Country: {country}")
    processed_urls = set()  # Keep track of URLs already queued in this run

    while await is_scan_running(redis):
        try:
            current_task_count = await get_active_tasks_count_from_redis(redis)
            if current_task_count >= settings.max_concurrent_tasks:
                logger.debug(f"Task limit ({settings.max_concurrent_tasks}) reached. Sleeping...")
                await asyncio.sleep(5)
                continue

            # Fetch URLs from random page
            page = randint(1, 100)
            new_urls = await fetch_camera_urls_from_source(session, country, page)

            queued_count = 0
            for cam_url in new_urls:
                if cam_url not in processed_urls:
                    # Check task limit again before queueing each task
                    if await get_active_tasks_count_from_redis(redis) >= settings.max_concurrent_tasks:
                        logger.debug("Task limit hit while queueing. Breaking loop for this page.")
                        break  # Stop queueing from this page if limit is hit

                    try:
                        task = celery.send_task(
                            "process_camera_task",
                            args=[cam_url, model_name],
                            task_id=f"scan-{uuid4().hex}",  # Use a unique ID prefix
                        )
                        await add_task_to_redis(task.id, redis)
                        processed_urls.add(cam_url)
                        queued_count += 1
                        logger.info(f"Queued task {task.id} for {cam_url}")
                    except (CeleryError, RedisError) as e:
                        logger.error(f"Failed to queue task or update Redis for {cam_url}: {e}", exc_info=True)
                        # Potentially break or implement retry logic
                        break  # Stop queueing if backend services fail

            logger.debug(f"Queued {queued_count} new tasks from page {page}.")
            await asyncio.sleep(1)  # Throttle requests to Insecam

        except ValueError as e:  # Catch specific errors like invalid country
            logger.error(f"Discovery loop error: {e}. Stopping scan.")
            await redis.delete(settings.scan_running_key)  # Stop the scan
            # Notify connected clients?
            await broadcast_message({"type": "scan_error", "message": str(e)})
            break
        except Exception as e:
            logger.error(f"Unexpected error in camera discovery loop: {e}", exc_info=True)
            await asyncio.sleep(10)  # Back off on unexpected errors

    logger.info("Camera discovery loop finished.")


# --- API Endpoints ---
@app.get("/", response_model=SuccessResponse, tags=["General"])
async def root():
    """Health check endpoint."""
    return SuccessResponse(message=f"{settings.app_name} is running.")


@app.get("/start-scan", response_model=SuccessResponse, tags=["Scanning"])
async def start_scan(
    redis: aioredis.Redis = Depends(get_redis_client),
    session: aiohttp.ClientSession = Depends(get_http_session),
    celery: Celery = Depends(get_celery),
    country: Optional[str] = Query(
        None, description="Optional two-letter country code (e.g., US, JP).", regex="^[A-Z]{2}$"
    ),
    model_name: Optional[str] = Query(
        settings.default_model, description=f"YOLO model name. Available: {', '.join(valid_models) or 'None'}"
    ),
) -> SuccessResponse:
    """Starts the background camera scanning process."""
    if await is_scan_running(redis):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Scan is already running.")

    # Validate model name
    if not valid_models:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No detection models available.")
    if model_name not in valid_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model name '{model_name}'. Available models: {', '.join(valid_models)}",
        )

    # Pre-validate country code roughly with Insecam (best effort)
    if country:
        try:
            logger.info(f"Validating country code '{country}' with Insecam...")
            # Do a quick HEAD request to check if the country page exists
            async with session.head(f"http://www.insecam.org/en/bycountry/{country}/") as resp:
                # Status < 400 usually means OK or redirect, 404 means not found
                if resp.status >= 400:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Country code '{country}' is not available on Insecam or the site is unreachable.",
                    )
                logger.info(f"Country code '{country}' seems valid.")
        except aiohttp.ClientError as e:
            logger.warning(f"Could not validate country code '{country}' due to network error: {e}")
            # Proceed anyway, but log the warning
        except HTTPException as e:
            raise e  # Re-raise validation error
        except Exception as e:
            logger.error(f"Unexpected error validating country code '{country}': {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Error validating country code."
            )

    try:
        await redis.set(settings.scan_running_key, 1)
        app_state["model_name"] = model_name  # Store globally selected model
        # Start the discovery loop as a background task
        asyncio.create_task(camera_discovery_loop(country, model_name, redis, session, celery))
        logger.info(f"Scan started. Country: {country or 'All'}, Model: {model_name}")
        return SuccessResponse(message="Scan started successfully.")
    except (RedisError, CeleryError) as e:
        logger.critical(f"Failed to start scan due to backend service error: {e}", exc_info=True)
        # Clean up scan running flag if set failed
        await redis.delete(settings.scan_running_key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to communicate with backend services to start scan.",
        )


@app.get("/stop-scan", response_model=SuccessResponse, tags=["Scanning"])
async def stop_scan(
    redis: aioredis.Redis = Depends(get_redis_client), celery: Celery = Depends(get_celery)
) -> SuccessResponse:
    """Stops the running scan and attempts to cancel pending tasks."""
    if not await is_scan_running(redis):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No scan is currently running.")

    logger.info("Stop scan request received. Initiating shutdown...")

    # 1. Delete the 'scan_running' flag in Redis first
    try:
        await redis.delete(settings.scan_running_key)
        logger.info("Scan running flag removed from Redis.")
    except RedisError as e:
        logger.error(f"Failed to remove scan running flag from Redis: {e}. Continuing stop process.", exc_info=True)
        # Continue stopping even if flag removal fails

    # 2. Get list of active tasks from Redis *before* clearing the set
    active_task_ids = await get_active_tasks_from_redis(redis)
    logger.info(f"Found {len(active_task_ids)} potentially active tasks in Redis set.")

    # 3. Clear the active tasks set in Redis
    try:
        await redis.delete(settings.active_tasks_key)
        logger.info(f"Cleared active tasks set '{settings.active_tasks_key}' in Redis.")
    except RedisError as e:
        logger.error(f"Failed to clear active tasks set in Redis: {e}. Continuing stop process.", exc_info=True)

    # 4. Attempt to revoke tasks in Celery (best effort)
    revoked_count = 0
    failed_revoke_count = 0
    if active_task_ids:
        logger.info(f"Attempting to revoke {len(active_task_ids)} Celery tasks...")
        # Using Celery's control is synchronous, consider alternatives if this blocks significantly
        # For simplicity, we use it directly here.
        try:
            # Use Celery control (synchronous call)
            control = celery.control
            for task_id in active_task_ids:
                try:
                    # Terminate=True sends SIGTERM to running tasks
                    control.revoke(task_id, terminate=True, signal="SIGTERM")
                    revoked_count += 1
                    logger.debug(f"Revocation request sent for task {task_id}")
                except Exception as e:  # Catch potential errors during revoke call
                    failed_revoke_count += 1
                    logger.warning(f"Failed to send revoke request for task {task_id}: {e}")
            logger.info(
                f"Finished sending revocation requests. Success: {revoked_count}, Failed: {failed_revoke_count}"
            )
        except CeleryError as e:
            logger.error(f"Celery control error during revocation: {e}", exc_info=True)
            # Continue the process despite control error

    # Notify clients that scan has stopped
    await broadcast_message({"type": "scan_stopped"})

    return SuccessResponse(message=f"Scan stop initiated. Attempted to revoke {revoked_count} tasks.")


@app.get("/scan-status", response_model=ScanStatusResponse, tags=["Scanning"])
async def scan_status(redis: aioredis.Redis = Depends(get_redis_client)) -> ScanStatusResponse:
    """Gets the current status of the scan."""
    running = await is_scan_running(redis)
    count = await get_active_tasks_count_from_redis(redis)
    # Ensure model_name is read from the shared app_state
    current_model = app_state.get("model_name", settings.default_model)
    return ScanStatusResponse(scan_running=running, active_tasks=count, selected_model=current_model)


# --- WebSocket Handling ---
async def broadcast_message(message: dict):
    """Sends a JSON message to all connected WebSocket clients."""
    disconnected_clients = set()
    message_str = str(message)  # For logging potentially large messages
    logger.debug(
        f"Broadcasting message to {len(connected_clients)} clients: {message_str[:200]}{'...' if len(message_str) > 200 else ''}"
    )

    # Use asyncio.gather for potentially faster broadcasting
    tasks = [client.send_json(message) for client in connected_clients]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for client, result in zip(connected_clients, results):
        if isinstance(result, Exception):
            # Handle specific disconnect exceptions vs other errors
            if isinstance(result, (WebSocketDisconnect, ConnectionResetError, RuntimeError)):
                logger.info(f"Client disconnected during broadcast: {result}")
            else:
                logger.error(f"Error sending message to client {client}: {result}", exc_info=result)
            disconnected_clients.add(client)

    # Remove disconnected clients efficiently
    if disconnected_clients:
        connected_clients.difference_update(disconnected_clients)
        logger.info(f"Removed {len(disconnected_clients)} disconnected clients. Remaining: {len(connected_clients)}")


async def websocket_result_listener(redis: aioredis.Redis, celery: Celery):
    """Periodically checks for completed Celery tasks and broadcasts results."""
    logger.info("Starting WebSocket result listener...")
    while True:
        try:
            active_task_ids = await get_active_tasks_from_redis(redis)
            if not active_task_ids:
                await asyncio.sleep(1)  # Sleep longer if no tasks are active
                continue

            tasks_to_remove = set()
            for task_id in active_task_ids:
                try:
                    # Use Celery's AsyncResult (this might involve I/O, consider optimizing if it becomes a bottleneck)
                    task_result = celery.AsyncResult(task_id)

                    if task_result.ready():
                        tasks_to_remove.add(task_id)  # Mark for removal regardless of state
                        if task_result.state == states.SUCCESS:
                            result_data = task_result.get()  # Can raise exception if task failed with one
                            if result_data and isinstance(result_data, dict) and result_data.get("processed"):
                                logger.debug(f"Broadcasting SUCCESS result for task {task_id}")
                                await broadcast_message({"type": "scan_result", "data": result_data})
                            elif result_data and isinstance(result_data, dict) and result_data.get("error"):
                                logger.debug(f"Broadcasting FAILED (but processed) result for task {task_id}")
                                await broadcast_message(
                                    {"type": "scan_result", "data": result_data}
                                )  # Send even if error occurred during processing
                            else:
                                logger.warning(
                                    f"Task {task_id} completed successfully but returned invalid data: {result_data}"
                                )
                        elif task_result.state == "SCAN_STOPPED":
                            logger.info(f"Task {task_id} was stopped during scan halt. Not broadcasting.")
                            # Already removed from Redis in the task itself or stop_scan
                        elif task_result.state == states.FAILURE:
                            logger.warning(f"Task {task_id} failed: {task_result.info}")
                            # Optionally broadcast a failure message
                            # await broadcast_message({"type": "task_failed", "task_id": task_id, "error": str(task_result.info)})
                            # Ensure it's removed from active set (should happen in task or stop_scan)
                            await remove_task_from_redis(task_id, redis)
                        elif task_result.state == states.REVOKED:
                            logger.info(f"Task {task_id} was revoked.")
                            # Ensure it's removed from active set
                            await remove_task_from_redis(task_id, redis)
                        else:
                            # Handle other states like PENDING, RETRY if necessary
                            logger.debug(f"Task {task_id} is in state {task_result.state}. Ignoring for now.")
                            tasks_to_remove.remove(task_id)  # Don't remove if not in a final state

                except CeleryError as e:
                    # Pssobility of retrying later, but for now just log and continue
                    logger.error(f"Celery error checking result for task {task_id}: {e}", exc_info=True)
                    tasks_to_remove.add(task_id)  # Remove to prevent repeated errors
                except Exception as e:
                    logger.error(f"Unexpected error processing task result for {task_id}: {e}", exc_info=True)
                    tasks_to_remove.add(task_id)  # Remove potentially problematic task ID

            # Batch remove processed/failed tasks from Redis
            if tasks_to_remove:
                logger.debug(f"Removing {len(tasks_to_remove)} tasks from active set: {tasks_to_remove}")
                # Use pipeline for efficiency if removing many tasks
                async with redis.pipeline(transaction=False) as pipe:
                    for task_id in tasks_to_remove:
                        pipe.srem(settings.active_tasks_key, task_id)
                    await pipe.execute()

            await asyncio.sleep(0.1)  # Short sleep to yield control and avoid busy-waiting

        except asyncio.CancelledError:
            logger.info("WebSocket result listener cancelled.")
            break
        except RedisError as e:
            logger.error(f"Redis error in result listener loop: {e}. Retrying after delay.", exc_info=True)
            await asyncio.sleep(5)  # Wait before retrying Redis operations
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket result listener: {e}", exc_info=True)
            await asyncio.sleep(5)  # Wait longer after unexpected errors


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    redis: aioredis.Redis = Depends(get_redis_client),  # Inject dependencies
    celery: Celery = Depends(get_celery),
):
    await websocket.accept()
    connected_clients.add(websocket)
    client_ip = websocket.client.host if websocket.client else "Unknown"
    logger.info(f"WebSocket client connected from {client_ip}. Total clients: {len(connected_clients)}")

    # Start the result listener task if it's not already running (simple check)
    # A more robust approach would use a shared flag or task registry
    listener_task_name = "websocket_result_listener_task"
    listener_task = next((task for task in asyncio.all_tasks() if task.get_name() == listener_task_name), None)
    if listener_task is None or listener_task.done():
        logger.info("No active result listener found, starting new one.")
        asyncio.create_task(websocket_result_listener(redis, celery), name=listener_task_name)
    else:
        logger.debug("Result listener already running.")

    try:
        # Keep the connection alive, listening for potential messages from client (e.g., commands)
        while True:
            # We don't expect messages from client in this design, but keep receive() for health check
            await websocket.receive_text()  # This will raise WebSocketDisconnect if client closes
    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_ip} disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection handler for {client_ip}: {e}", exc_info=True)
        # Try to close gracefully
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass  # Ignore errors during close after another error
    finally:
        connected_clients.remove(websocket)
        logger.info(f"WebSocket client removed. Total clients: {len(connected_clients)}")
        # Consider stopping the listener task if no clients are connected (add logic if needed)
        # if not connected_clients and listener_task and not listener_task.done():
        #     logger.info("Last client disconnected, stopping result listener.")
        #     listener_task.cancel()
