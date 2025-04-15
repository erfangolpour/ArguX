import { OrbitControls } from "@react-three/drei";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import React, {
	useCallback,
	useEffect,
	useMemo,
	useRef,
	useState,
} from "react";
import {
	BufferGeometry,
	Color,
	DirectionalLight,
	Float32BufferAttribute,
	Fog,
	MeshPhongMaterial,
	Object3D,
	PerspectiveCamera,
	Points,
	PointsMaterial,
	Scene,
	Spherical,
	Vector3,
} from "three";
import ThreeGlobe from "three-globe";
import { OrbitControls as OrbitControlsType } from "three-stdlib";
import { useCameraStore } from "../../store/cameraStore";
import { Camera as CameraType } from "../../types/types";
import { CameraTooltip } from "./CameraTooltip";
import countries from "./countries.json";

const GLOBE_RADIUS = 100;
const POINTS_RADIUS = GLOBE_RADIUS + 1;
const POINT_BASE_SIZE = 5;
const RAYCASTER_THROTTLE_MS = 50; // Throttle raycasting to every 50ms

// Colors for different camera states
const CAMERA_COLORS = {
	normal: "#38bdf8", // Default blue color for normal cameras
	error: "#ef4444", // Red color for cameras with errors
	detected: "#22c55e", // Green color for cameras with detections
};

// Helper Functions
const latLngToVector3 = (lat: number, lng: number, radius: number): Vector3 => {
	const phi = Math.PI * (0.5 - lat / 180);
	const theta = Math.PI * (lng / 180);
	const spherical = new Spherical(radius, phi, theta);
	return new Vector3().setFromSpherical(spherical);
};

// Create a custom point material to match the previous version
const createPointsMaterial = () => {
	return new PointsMaterial({
		size: POINT_BASE_SIZE,
		vertexColors: true, // Enable vertex colors
		sizeAttenuation: false,
	});
};

// Throttle function to limit execution frequency
function throttle<T extends (...args: any[]) => any>(
	callback: T,
	delay: number,
): (...args: Parameters<T>) => void {
	let lastCall = 0;
	return (...args: Parameters<T>) => {
		const now = Date.now();
		if (now - lastCall >= delay) {
			lastCall = now;
			callback(...args);
		}
	};
}

// GlobeInstance - purely Three.js content without any HTML/DOM elements
const GlobeInstance: React.FC<{
	onMouseMove: (x: number, y: number) => void;
	onHover: (cameraId: number | null) => void;
	onClick: () => void;
}> = ({ onMouseMove, onHover, onClick }) => {
	// Use individual primitive selectors
	const filteredCameras = useCameraStore((state) => state.filteredCameras);
	const selectedCamera = useCameraStore((state) => state.selectedCamera);

	const globeRef = useRef<ThreeGlobe>(null!);
	const pointsRef = useRef<Points>(null!);
	const globeGroupRef = useRef<Object3D>(null!); // Ref for the group containing the globe

	// Keep track of if a camera is selected to prevent auto-rotation
	const selectedCameraRef = useRef<boolean>(false);

	// Map point indices to camera data for raycasting
	const [cameraPointsMap, setCameraPointsMap] = useState<CameraType[]>([]);

	// Track loading and animation state
	const [isGlobeLoaded, setIsGlobeLoaded] = useState<boolean>(false);
	const isInitialLoadRef = useRef<boolean>(true);

	// Local state for hover tracking
	const [hoveredPointIndex, setHoveredPointIndex] = useState<number | null>(
		null,
	);

	// Three.js setup from context
	const { scene, camera, gl, raycaster, mouse, size } = useThree();
	const orbitControlsRef = useRef<OrbitControlsType>(null!);

	// Track mouse hover state over the globe container
	const [isGlobeHovered, setIsGlobeHovered] = useState(false);

	// Add a container ref for click handling
	const containerRef = useRef<Object3D>(null!);

	// Object pool for reusing point geometries and materials
	const geometryPoolRef = useRef<BufferGeometry | null>(null);
	const materialPoolRef = useRef<PointsMaterial | null>(null);

	// Update selectedCameraRef when selectedCamera changes
	useEffect(() => {
		selectedCameraRef.current = !!selectedCamera;

		// When a camera is selected, always disable auto-rotation
		if (selectedCamera && orbitControlsRef.current) {
			orbitControlsRef.current.autoRotate = false;
		}
	}, [selectedCamera]);

	// Initialize Globe and Points only once
	useEffect(() => {
		console.log("Initializing Globe and Points...");

		// Create a group for the globe and its components for animation
		const globeGroup = new Object3D();
		globeGroup.visible = false; // Start hidden for initial animation
		globeGroupRef.current = globeGroup;
		scene.add(globeGroup);

		// Create the globe
		const globe = new ThreeGlobe()
			.hexPolygonsData(countries.features)
			.hexPolygonResolution(3)
			.hexPolygonMargin(0.7)
			.hexPolygonColor(() => "rgba(255,255,255,0.7)")
			.showAtmosphere(true)
			.atmosphereColor("#FFFFFF")
			.atmosphereAltitude(0.1);

		// Customize globe material to match previous version
		const globeMaterial = globe.globeMaterial() as MeshPhongMaterial;
		globeMaterial.color.set("#062056");
		globeMaterial.emissive = new Color("#062056");
		globeMaterial.emissiveIntensity = 0.1;
		globeMaterial.shininess = 0.9;

		globeRef.current = globe;
		globeGroup.add(globe);

		// Initialize object pool
		if (!geometryPoolRef.current) {
			geometryPoolRef.current = new BufferGeometry();
		}
		if (!materialPoolRef.current) {
			materialPoolRef.current = createPointsMaterial();
		}

		// Create points with material from the pool
		const points = new Points(
			geometryPoolRef.current,
			materialPoolRef.current,
		);
		points.renderOrder = 999; // Ensure points render on top of everything
		points.frustumCulled = false; // Keep points visible even when globe rotates

		pointsRef.current = points;
		globeGroup.add(points);

		// Consolidated initialization and animation in a single timeout
		const initTimer = setTimeout(() => {
			setIsGlobeLoaded(true);
		}, 300);

		return () => {
			console.log("Cleaning up Globe and Points...");
			clearTimeout(initTimer);
			if (globeGroupRef.current) scene.remove(globeGroupRef.current);

			// Don't dispose pooled objects, they'll be reused
		};
	}, [scene]);

	// Handle globe entrance animation
	useEffect(() => {
		if (isGlobeLoaded && globeGroupRef.current) {
			// Make the globe visible first
			globeGroupRef.current.visible = true;

			// Apply initial scale for animation
			globeGroupRef.current.scale.set(0.7, 0.7, 0.7); // Start slightly smaller
			globeGroupRef.current.rotation.y = -Math.PI; // Start rotated

			// Initial opacity for atmosphere and globe
			if (globeRef.current) {
				const material =
					globeRef.current.globeMaterial() as MeshPhongMaterial;
				material.opacity = 0;
				material.transparent = true;
			}

			// Animation timing
			const startTime = performance.now();
			const duration = 1500; // 1.5 seconds fade in

			// Animate function
			const animate = () => {
				const now = performance.now();
				const elapsed = now - startTime;
				const progress = Math.min(elapsed / duration, 1);

				// Ease in cubic function
				const eased = 1 - Math.pow(1 - progress, 3);

				if (globeGroupRef.current) {
					// Scale from 0.7 to 1.0
					const scale = 0.7 + 0.3 * eased;
					globeGroupRef.current.scale.set(scale, scale, scale);

					// Rotate slightly as it appears
					globeGroupRef.current.rotation.y =
						-Math.PI + Math.PI * eased;

					// Fade in the opacity
					if (globeRef.current) {
						const material =
							globeRef.current.globeMaterial() as MeshPhongMaterial;
						material.opacity = eased;
					}
				}

				if (progress < 1) {
					requestAnimationFrame(animate);
				} else {
					// Animation complete - clean up
					if (globeRef.current) {
						const material =
							globeRef.current.globeMaterial() as MeshPhongMaterial;
						material.opacity = 1;
						material.transparent = false; // No longer need transparency
					}
					isInitialLoadRef.current = false;
				}
			};

			// Start animation
			if (isInitialLoadRef.current) {
				requestAnimationFrame(animate);
			}
		}
	}, [isGlobeLoaded]);

	// Update points when filteredCameras change - reuse buffer geometries
	useEffect(() => {
		if (!pointsRef.current) return;

		// Create positions for each camera point
		const positions: number[] = [];
		const colors: number[] = []; // Add color array for each point

		// Store cameras that correspond to points for raycasting later
		const pointsMap: CameraType[] = [];

		filteredCameras.forEach((camera) => {
			// Calculate position on globe
			const position = latLngToVector3(
				camera.geo.lat ?? 0,
				camera.geo.lon ?? 0,
				POINTS_RADIUS, // Consistent radius
			);

			positions.push(position.x, position.y, position.z);

			// Determine point color based on camera error status
			const colorHex = camera.error
				? CAMERA_COLORS.error
				: camera.detections.length > 0
					? CAMERA_COLORS.detected
					: CAMERA_COLORS.normal;
			const color = new Color(colorHex);
			colors.push(color.r, color.g, color.b);

			// Add camera to map
			pointsMap.push(camera);
		});

		// If no cameras, set empty arrays
		if (positions.length === 0) {
			positions.push(0, 0, 0); // Add a dummy point at center
			colors.push(0, 0, 0); // Black
		}

		// Reuse existing geometry from pool if possible, otherwise create new attributes
		if (geometryPoolRef.current) {
			// Update position attribute
			geometryPoolRef.current.setAttribute(
				"position",
				new Float32BufferAttribute(positions, 3),
			);

			// Update color attribute
			geometryPoolRef.current.setAttribute(
				"color",
				new Float32BufferAttribute(colors, 3),
			);

			// Update bounding sphere for raycasting
			geometryPoolRef.current.computeBoundingSphere();
		}

		// Store the mapping between point indices and cameras
		setCameraPointsMap(pointsMap);

		// Update point count to manage visibility
		if (pointsRef.current) {
			// Reset previous point indices
			setHoveredPointIndex(null);
			onHover(null);
		}
	}, [filteredCameras, onHover]);

	// Handle mouse move for pointer tracking - and pass coordinates to parent
	useEffect(() => {
		const handleCanvasMouseMove = () => {
			// Forward normalized coordinates for tooltip positioning in the DOM
			onMouseMove(
				(mouse.x * 0.5 + 0.5) * size.width,
				(-mouse.y * 0.5 + 0.5) * size.height,
			);
		};

		// Add listener to the canvas
		const canvas = gl.domElement;
		canvas.addEventListener("mousemove", handleCanvasMouseMove);

		return () => {
			canvas.removeEventListener("mousemove", handleCanvasMouseMove);
		};
	}, [gl, mouse, onMouseMove, size]);

	// Handle pointer enter/leave to control auto-rotation
	useEffect(() => {
		const canvas = gl.domElement;

		const handlePointerEnter = () => {
			setIsGlobeHovered(true);
			if (orbitControlsRef.current) {
				orbitControlsRef.current.autoRotate = false;
			}
		};

		const handlePointerLeave = () => {
			setIsGlobeHovered(false);
			// Only resume auto-rotation if no tooltip is visible
			// AND no camera is selected (checking selectedCameraRef)
			if (
				orbitControlsRef.current &&
				hoveredPointIndex === null &&
				!selectedCameraRef.current
			) {
				orbitControlsRef.current.autoRotate = true;
			}
		};

		canvas.addEventListener("pointerenter", handlePointerEnter);
		canvas.addEventListener("pointerleave", handlePointerLeave);

		return () => {
			canvas.removeEventListener("pointerenter", handlePointerEnter);
			canvas.removeEventListener("pointerleave", handlePointerLeave);
		};
	}, [gl, hoveredPointIndex]);

	// Create a click handler that works with the scene
	useEffect(() => {
		const handleClick = () => {
			if (hoveredPointIndex !== null) {
				onClick();
			}
		};

		// Add listener to the canvas
		const canvas = gl.domElement;
		canvas.addEventListener("click", handleClick);

		return () => {
			canvas.removeEventListener("click", handleClick);
		};
	}, [hoveredPointIndex, onClick, gl]);

	// Throttled raycasting handler
	const handleRaycast = useCallback(
		throttle(() => {
			if (
				!pointsRef.current ||
				cameraPointsMap.length === 0 ||
				!orbitControlsRef.current?.enabled
			) {
				if (hoveredPointIndex !== null) {
					setHoveredPointIndex(null);
					onHover(null);
				}
				return;
			}

			raycaster.setFromCamera(mouse, camera);
			raycaster.params.Points = { threshold: 1 }; // Adjust threshold for easier hover

			const intersects = raycaster.intersectObject(pointsRef.current);

			if (intersects.length > 0) {
				const index = intersects[0].index;
				if (
					index !== undefined &&
					index >= 0 &&
					index < cameraPointsMap.length
				) {
					if (hoveredPointIndex !== index) {
						setHoveredPointIndex(index);
						onHover(index);

						// Stop auto-rotation when hovering over a point
						if (orbitControlsRef.current) {
							orbitControlsRef.current.autoRotate = false;
						}
					}
				}
			} else if (hoveredPointIndex !== null) {
				setHoveredPointIndex(null);
				onHover(null);

				// Resume auto-rotation when not hovering over a point,
				// but only if not hovering the globe itself
				// AND no camera is selected (checking selectedCameraRef)
				if (
					orbitControlsRef.current &&
					!isGlobeHovered &&
					!selectedCameraRef.current
				) {
					orbitControlsRef.current.autoRotate = true;
				}
			}
		}, RAYCASTER_THROTTLE_MS),
		[
			raycaster,
			mouse,
			camera,
			pointsRef,
			cameraPointsMap,
			hoveredPointIndex,
			orbitControlsRef,
			isGlobeHovered,
			selectedCameraRef,
			onHover,
		],
	);

	// Handle raycasting for hover effects using useFrame but throttled
	useFrame(() => {
		// Only perform raycasting at throttled intervals
		handleRaycast();
	});

	// Handle camera selection - animate to selected camera
	useEffect(() => {
		if (!selectedCamera || !orbitControlsRef.current || !camera) return;

		const controls = orbitControlsRef.current;
		controls.enabled = false;
		controls.autoRotate = false;

		const pointPosition = latLngToVector3(
			selectedCamera.geo.lat ?? 0,
			selectedCamera.geo.lon ?? 0,
			POINTS_RADIUS,
		);

		// Keep the orbit controls target at the center of the globe
		const centerTarget = new Vector3(0, 0, 0);
		controls.target.copy(centerTarget);

		// Get current camera position and rotation state
		const startPosition = camera.position.clone();

		// Get direction vectors for interpolation
		const startDirection = startPosition.clone().normalize();
		const endDirection = pointPosition.clone().normalize();

		// Calculate final camera position (same direction as target point but at proper distance)
		const cameraDistance = 250; // Good viewing distance
		const finalCameraPosition = endDirection
			.clone()
			.multiplyScalar(cameraDistance);

		// Animation duration and timing
		const duration = 1200; // 1.2 seconds
		const startTime = performance.now();

		// Custom easing function (cubic out)
		const easeOutCubic = (t: number): number => {
			return 1 - Math.pow(1 - t, 3);
		};

		// Animation loop
		function animate() {
			const elapsed = performance.now() - startTime;
			const progress = Math.min(elapsed / duration, 1);
			const easedProgress = easeOutCubic(progress);

			// Create intermediate direction by properly interpolating between start and end directions
			const currentDirection = new Vector3();
			currentDirection.copy(startDirection);
			currentDirection.lerp(endDirection, easedProgress).normalize();

			// Calculate intermediate camera position along the arc
			const distanceProgression =
				startPosition.length() +
				(cameraDistance - startPosition.length()) * easedProgress;
			const newPosition = currentDirection
				.clone()
				.multiplyScalar(distanceProgression);

			// Apply the new camera position
			camera.position.copy(newPosition);

			// Make camera look at the target point while maintaining up direction
			camera.lookAt(pointPosition);

			// Keep the target at the center for rotation purposes
			controls.target.copy(centerTarget);
			controls.update();

			// Continue animation if not complete
			if (progress < 1) {
				return requestAnimationFrame(animate);
			}

			// Ensure we're perfectly aligned at the end
			camera.position.copy(finalCameraPosition);
			camera.lookAt(pointPosition);
			controls.update();
			controls.enabled = true;

			return null;
		}

		// Start the animation
		const animationId = requestAnimationFrame(animate);

		// Clean up animation if component unmounts during animation
		return () => {
			if (animationId !== null) {
				cancelAnimationFrame(animationId);
			}
			controls.enabled = true;
		};
	}, [selectedCamera, camera]);

	return (
		<>
			{/* Use a group as a container for event handling */}
			<group ref={containerRef}>
				<OrbitControls
					ref={orbitControlsRef}
					enablePan={false}
					enableZoom={true}
					autoRotate={!selectedCameraRef.current} // Initially only auto-rotate if no camera is selected
					autoRotateSpeed={0.5} // Match previous speed
					minDistance={GLOBE_RADIUS * 1.5}
					maxDistance={GLOBE_RADIUS * 4.0}
					target={[0, 0, 0]}
					makeDefault
				/>
			</group>
		</>
	);
};

// Main Exported Component with proper separation between Canvas and DOM
export const Globe = () => {
	// Scene and camera setup
	const { scene, camera } = useMemo(() => {
		const scene = new Scene();
		scene.fog = new Fog(0xffffff, 400, 2000); // Match previous fog

		const camera = new PerspectiveCamera(
			45,
			window.innerWidth / window.innerHeight,
			1,
			2000,
		);
		camera.position.z = GLOBE_RADIUS * 3.5;

		// Add light to camera like in previous version
		const followLight = new DirectionalLight(0xffffff, 1);
		followLight.position.set(-200, 500, 200);
		camera.add(followLight);
		scene.add(camera);

		return { scene, camera };
	}, []);

	// State for the tooltip
	const [tooltipVisible, setTooltipVisible] = useState(false);
	const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
	const [displayBelow, setDisplayBelow] = useState(false);
	const [hoveredInstanceId, setHoveredInstanceId] = useState<number | null>(
		null,
	);

	// State for tracking loading
	const [globeLoading, setGlobeLoading] = useState(true);

	// Get hovered camera from store based on index
	const hoveredCamera = useCameraStore(
		useCallback(
			(state) => {
				if (hoveredInstanceId === null) return null;

				// Find the camera by scanning through filtered cameras
				const cameras = state.filteredCameras;
				if (
					hoveredInstanceId >= 0 &&
					hoveredInstanceId < cameras.length
				) {
					return cameras[hoveredInstanceId];
				}
				return null;
			},
			[hoveredInstanceId],
		),
	);

	// Handler for mouse movement from Canvas - simplified
	const handleMouseMove = useCallback((x: number, y: number) => {
		// Calculate tooltip position based on screen position
		const displayBelow = y < window.innerHeight / 2;
		setTooltipPosition({ x, y });
		setDisplayBelow(displayBelow);
	}, []);

	// Handler for hover state from Canvas
	const handleHover = useCallback((instanceId: number | null) => {
		setHoveredInstanceId(instanceId);
		setTooltipVisible(instanceId !== null);
	}, []);

	// Handler for click from Canvas
	const handleClick = useCallback(() => {
		if (hoveredCamera) {
			useCameraStore.getState().setSelectedCamera(hoveredCamera);
			setHoveredInstanceId(null);
			setTooltipVisible(false);
		}
	}, [hoveredCamera]);

	useEffect(() => {
		const handleResize = () => {
			camera.aspect = window.innerWidth / window.innerHeight;
			camera.updateProjectionMatrix();
		};
		window.addEventListener("resize", handleResize);

		// Hide loading indicator after a short delay to ensure smooth transition
		const loadingTimer = setTimeout(() => {
			setGlobeLoading(false);
		}, 1000);

		return () => {
			window.removeEventListener("resize", handleResize);
			clearTimeout(loadingTimer);
		};
	}, [camera]);

	return (
		<div className="relative size-full cursor-grab active:cursor-grabbing">
			{/* Loading overlay with fade-out effect */}
			{globeLoading && (
				<div
					className="bg-opacity-70 absolute inset-0 z-10 flex items-center justify-center transition-opacity duration-500"
					style={{ opacity: globeLoading ? 1 : 0 }}
				>
					<div className="flex flex-col items-center space-y-3">
						<div className="h-10 w-10 animate-spin rounded-full border-4 border-white border-t-transparent"></div>
						<div className="font-medium text-white">
							Loading Globe...
						</div>
					</div>
				</div>
			)}

			<Canvas
				scene={scene}
				camera={camera}
				gl={{ antialias: true }}
				className="bg-radial from-blue-900 to-black" // Match previous background
			>
				{/* Match previous lighting */}
				<ambientLight color="#38bdf8" intensity={0.6} />
				<GlobeInstance
					onMouseMove={handleMouseMove}
					onHover={handleHover}
					onClick={handleClick}
				/>
			</Canvas>

			{/* Tooltip is outside Canvas and part of the DOM */}
			<CameraTooltip
				camera={hoveredCamera}
				visible={tooltipVisible}
				position={tooltipPosition}
				displayBelow={displayBelow}
			/>
		</div>
	);
};
