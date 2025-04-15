import { create } from "zustand";
import {
	AppCamera,
	CameraScanResult,
	CountryCount,
	ErrorResponse,
	Geo,
	ObjectCount,
	ScanStatusResponse,
	SuccessResponse,
} from "../types/types"; // Import consolidated types

// Constants for WebSocket reconnection
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 3000; // Initial delay 3s

// Default Geo object for cameras with missing data
const defaultGeo: Geo = {
	country: "Unknown",
	countryCode: "XX",
	region: "Unknown",
	regionName: "Unknown",
	city: "Unknown",
	zip: "N/A",
	lat: 0,
	lon: 0,
	timezone: "Unknown",
	isp: "Unknown",
	org: "Unknown",
	as_info: "Unknown",
	query: "Unknown",
};

// Camera type filter options
export type CameraTypeFilter = "all" | "errors" | "empty" | "detected";

interface CameraState {
	// Camera data
	cameras: AppCamera[];
	cameraIdMap: Map<string, AppCamera>; // Use Map for faster lookups/updates

	// Selected camera
	selectedCamera: AppCamera | null;
	setSelectedCamera: (camera: AppCamera | null) => void;

	// Filtered cameras (derived state, calculated on demand or memoized)
	filteredCameras: AppCamera[];
	filterCameras: () => void; // Trigger filtering
	clearFilters: () => void;

	// Search term
	searchTerm: string;
	setSearchTerm: (term: string) => void;

	// Camera type filter
	cameraTypeFilter: CameraTypeFilter;
	setCameraTypeFilter: (type: CameraTypeFilter) => void;

	// Object and country filters
	selectedObjects: string[];
	toggleObjectSelection: (objectName: string) => void;
	selectedCountries: string[];
	toggleCountrySelection: (countryName: string) => void;

	// Statistics (can be maps for efficiency, converted to arrays for UI)
	objectCountMap: Map<string, number>;
	getObjectCounts: () => ObjectCount[]; // Selector function
	countryCountMap: Map<string, number>;
	getCountryCounts: () => CountryCount[]; // Selector function

	// Scan state
	scanRunning: boolean;
	scanModel: string; // Store the model used for the current/last scan
	activeTasks: number;
	isLoading: boolean; // General loading state for API calls
	error: string | null; // Store user-friendly error messages
	clearError: () => void; // Action to clear the error

	// API Actions
	startScan: (
		country?: string | null,
		modelName?: string | null,
	) => Promise<void>;
	stopScan: () => Promise<void>;
	checkScanStatus: () => Promise<void>;

	// WebSocket
	ws: WebSocket | null;
	connectWebSocket: () => void;
	disconnectWebSocket: () => void;
	isConnecting: boolean; // Track connection attempt state
	reconnectAttempts: number; // Track reconnection attempts
}

export const useCameraStore = create<CameraState>((set, get) => ({
	// Initial State
	cameras: [],
	cameraIdMap: new Map<string, AppCamera>(),
	selectedCamera: null,
	filteredCameras: [],
	searchTerm: "",
	cameraTypeFilter: "all",
	selectedObjects: [],
	selectedCountries: [],
	objectCountMap: new Map<string, number>(),
	countryCountMap: new Map<string, number>(),
	scanRunning: false,
	scanModel: "yolo11n.pt", // Default model
	activeTasks: 0,
	isLoading: false,
	error: null,
	ws: null,
	isConnecting: false,
	reconnectAttempts: 0,

	// --- Actions ---
	clearError: () => set({ error: null }),

	setSelectedCamera: (camera: AppCamera | null) =>
		set({ selectedCamera: camera }),

	setSearchTerm: (term: string) => {
		set({ searchTerm: term });
		get().filterCameras(); // Re-filter when search term changes
	},

	setCameraTypeFilter: (type: CameraTypeFilter) => {
		set({ cameraTypeFilter: type });
		get().filterCameras(); // Re-filter when camera type filter changes
	},

	toggleObjectSelection: (objectName: string) => {
		set((state) => {
			const newSelectedObjects = new Set(state.selectedObjects);
			if (newSelectedObjects.has(objectName)) {
				newSelectedObjects.delete(objectName);
			} else {
				newSelectedObjects.add(objectName);
			}
			return { selectedObjects: Array.from(newSelectedObjects) };
		});
		get().filterCameras();
	},

	toggleCountrySelection: (countryName: string) => {
		set((state) => {
			const newSelectedCountries = new Set(state.selectedCountries);
			if (newSelectedCountries.has(countryName)) {
				newSelectedCountries.delete(countryName);
			} else {
				newSelectedCountries.add(countryName);
			}
			return { selectedCountries: Array.from(newSelectedCountries) };
		});
		get().filterCameras();
	},

	clearFilters: () => {
		set({
			selectedObjects: [],
			selectedCountries: [],
			searchTerm: "", // Also clear search term
			cameraTypeFilter: "all", // Also reset camera type filter
		});
		get().filterCameras();
	},

	// --- Filtering Logic ---
	filterCameras: () => {
		const { cameras, selectedObjects, selectedCountries, searchTerm, cameraTypeFilter } =
			get();
		const term = searchTerm.trim().toLowerCase();
		const hasObjectFilter = selectedObjects.length > 0;
		const hasCountryFilter = selectedCountries.length > 0;
		const hasSearchFilter = term.length > 0;
		const hasCameraTypeFilter = cameraTypeFilter !== "all";

		// If no filters, return all cameras
		if (!hasObjectFilter && !hasCountryFilter && !hasSearchFilter && !hasCameraTypeFilter) {
			set({ filteredCameras: cameras });
			return;
		}

		const filtered = cameras.filter((camera) => {
			// Check object filter (must match AT LEAST ONE selected object if filter applied)
			const matchesObject =
				!hasObjectFilter ||
				selectedObjects.some((obj) =>
					camera.detections.some((det) => det.label === obj),
				);

			// Check country filter (must match AT LEAST ONE selected country if filter applied)
			const matchesCountry =
				!hasCountryFilter ||
				(camera.geo.countryCode &&
					selectedCountries.includes(camera.geo.countryCode));

			// Check search term (matches various fields)
			const matchesSearch =
				!hasSearchFilter ||
				(camera.geo.query &&
					camera.geo.query.toLowerCase().includes(term)) ||
				(camera.geo.city &&
					camera.geo.city.toLowerCase().includes(term)) ||
				(camera.geo.regionName &&
					camera.geo.regionName.toLowerCase().includes(term)) ||
				(camera.geo.country &&
					camera.geo.country.toLowerCase().includes(term)) ||
				(camera.geo.org &&
					camera.geo.org.toLowerCase().includes(term)) ||
				(camera.geo.as_info &&
					camera.geo.as_info.toLowerCase().includes(term)) ||
				camera.detections.some((det) =>
					det.label.toLowerCase().includes(term),
				);
			
			// Check camera type filter
			const matchesCameraType = !hasCameraTypeFilter || (
				(cameraTypeFilter === "errors" && camera.error && !camera.processed) ||
				(cameraTypeFilter === "empty" && camera.processed && camera.detections.length === 0) ||
				(cameraTypeFilter === "detected" && camera.processed && camera.detections.length > 0)
			);

			// Must match all active filters
			return matchesObject && matchesCountry && matchesSearch && matchesCameraType;
		});

		set({ filteredCameras: filtered });
	},

	// --- Statistics Selectors ---
	getObjectCounts: () => {
		const { objectCountMap } = get();
		return Array.from(objectCountMap.entries())
			.map(([name, count]) => ({ name, count }))
			.sort((a, b) => b.count - a.count); // Keep default sort by count desc
	},

	getCountryCounts: () => {
		const { countryCountMap } = get();
		return Array.from(countryCountMap.entries())
			.map(([name, count]) => ({ name, count }))
			.sort((a, b) => b.count - a.count); // Keep default sort by count desc
	},

	// --- WebSocket Logic ---
	connectWebSocket: () => {
		const { ws, scanRunning, isConnecting } = get();

		// Prevent multiple connection attempts or connecting if scan isn't running
		if (
			(ws && ws.readyState === WebSocket.OPEN) ||
			isConnecting ||
			!scanRunning
		) {
			if (!scanRunning)
				console.log("Scan not running, WebSocket connection skipped.");
			if (isConnecting)
				console.log(
					"WebSocket connection attempt already in progress.",
				);
			return;
		}

		set({ isConnecting: true, error: null }); // Clear previous errors on new attempt
		console.log("Attempting WebSocket connection...");

		const newWs = new WebSocket(`ws://${window.location.hostname}:8000/ws`); // Use dynamic hostname

		newWs.onopen = () => {
			console.log("WebSocket connected successfully.");
			set({
				ws: newWs,
				isConnecting: false,
				reconnectAttempts: 0,
				error: null,
			}); // Reset attempts on success
		};

		newWs.onmessage = (event) => {
			try {
				const message = JSON.parse(event.data);

				// Handle different message types from backend
				switch (message.type) {
					case "scan_result":
						const rawResult: CameraScanResult = message.data;

						// Ignore results that failed processing entirely (unless you want to show them)
						// Also ignore if we somehow already have this task ID (shouldn't happen with UUIDs)
						if (!rawResult || get().cameraIdMap.has(rawResult.id)) {
							return;
						}

						// Transform backend result to frontend AppCamera type
						const camera: AppCamera = {
							...rawResult,
							timestamp: new Date(rawResult.timestamp), // Parse ISO string
							geo: rawResult.geo
								? { ...defaultGeo, ...rawResult.geo }
								: defaultGeo, // Merge with defaults
						};

						// Update state immutably
						set((state) => {
							const newCameraIdMap = new Map(state.cameraIdMap);
							newCameraIdMap.set(camera.id, camera);

							const newCameras = Array.from(
								newCameraIdMap.values(),
							);

							// Update counts efficiently using the new camera data
							const newObjectCountMap = new Map(
								state.objectCountMap,
							);
							camera.detections.forEach((detection) => {
								newObjectCountMap.set(
									detection.label,
									(newObjectCountMap.get(detection.label) ||
										0) + 1,
								);
							});

							const newCountryCountMap = new Map(
								state.countryCountMap,
							);
							if (camera.geo.countryCode) {
								// Ensure country code exists
								newCountryCountMap.set(
									camera.geo.countryCode,
									(newCountryCountMap.get(
										camera.geo.countryCode,
									) || 0) + 1,
								);
							}

							return {
								cameras: newCameras,
								cameraIdMap: newCameraIdMap,
								objectCountMap: newObjectCountMap,
								countryCountMap: newCountryCountMap,
							};
						});

						// Trigger filtering after state update
						get().filterCameras();
						break;

					case "scan_stopped":
						console.log(
							"Received scan_stopped message from backend.",
						);
						set({
							scanRunning: false,
							activeTasks: 0,
							// Add a success message to inform user
							error: "Scan completed successfully.",
						});
						// No need to close WS here, backend manages lifecycle
						break;

					case "scan_error":
						console.error(
							"Received scan_error message from backend:",
							message.message,
						);
						set({
							scanRunning: false,
							activeTasks: 0,
							error: `Scan failed: ${message.message || "Unknown error"}`,
						});
						get().disconnectWebSocket(); // Disconnect on fatal scan error
						break;

					// Add other message types if needed

					default:
						console.warn(
							"Received unknown WebSocket message type:",
							message.type,
						);
				}
			} catch (e) {
				console.error(
					"Error processing WebSocket message:",
					e,
					event.data,
				);
				// Update UI state with error message
				set({
					error: `Failed to process server message: ${e instanceof Error ? e.message : "Unknown error"}`,
				});
			}
		};

		newWs.onclose = (event) => {
			console.log(
				`WebSocket closed. Code: ${event.code}, Reason: ${event.reason}`,
			);
			// Only attempt reconnect if closed unexpectedly and scan should be running
			if (
				event.code !== 1000 &&
				get().scanRunning &&
				get().reconnectAttempts < MAX_RECONNECT_ATTEMPTS
			) {
				set((state) => ({
					ws: null,
					isConnecting: false,
					reconnectAttempts: state.reconnectAttempts + 1,
				}));
				const delay =
					RECONNECT_DELAY_MS *
					Math.pow(2, get().reconnectAttempts - 1); // Exponential backoff
				console.log(
					`Attempting WebSocket reconnect in ${delay / 1000}s (Attempt ${get().reconnectAttempts})...`,
				);
				setTimeout(() => {
					if (get().scanRunning) {
						// Double check if scan is still supposed to be running
						get().connectWebSocket();
					} else {
						console.log(
							"Scan stopped while waiting to reconnect. Aborting reconnect.",
						);
						set({ reconnectAttempts: 0 }); // Reset attempts
					}
				}, delay);
			} else {
				if (event.code === 1000) {
					console.log("WebSocket closed normally.");
				} else if (get().reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
					console.error(
						"WebSocket reconnect failed after multiple attempts.",
					);
					set({
						error: "Connection to server lost. Please restart scan if needed.",
					});
				} else if (!get().scanRunning) {
					console.log("WebSocket closed because scan stopped.");
				}
				set({ ws: null, isConnecting: false, reconnectAttempts: 0 }); // Clear state if not reconnecting
			}
		};

		newWs.onerror = (error) => {
			console.error("WebSocket error:", error);
			set({
				ws: null, // Ensure ws is null on error
				isConnecting: false,
				error: "WebSocket connection error. Check console for details.",
			});
			// onclose will likely be called after onerror, handling reconnect logic
		};
	},

	disconnectWebSocket: () => {
		const { ws } = get();
		if (ws) {
			console.log("Disconnecting WebSocket manually.");
			ws.close(1000, "Client requested disconnect"); // Use code 1000 for normal closure
		}
		set({ ws: null, isConnecting: false, reconnectAttempts: 0 });
	},

	// --- API Actions ---
	startScan: async (country?: string | null, modelName?: string | null) => {
		set({
			isLoading: true,
			error: null,
			cameras: [],
			cameraIdMap: new Map(),
			selectedCamera: null,
			objectCountMap: new Map(),
			countryCountMap: new Map(),
			selectedObjects: [],
			selectedCountries: [],
			filteredCameras: [],
		}); // Reset state on new scan

		const params = new URLSearchParams();
		if (country) params.append("country", country);
		if (modelName) params.append("model_name", modelName); // Use 'model_name' as per backend Query param name

		try {
			const response = await fetch(
				`http://localhost:8000/start-scan?${params.toString()}`,
			);
			const result: SuccessResponse | ErrorResponse =
				await response.json();

			if (response.ok && result.status === "success") {
				set({
					scanRunning: true,
					scanModel: modelName || get().scanModel,
					isLoading: false,
				});
				get().connectWebSocket(); // Connect WS only after successful start
			} else {
				const errorDetail =
					(result as ErrorResponse).detail ||
					`HTTP ${response.status}`;
				const errorMessage =
					(result as ErrorResponse).message || "Failed to start scan";
				throw new Error(`${errorMessage}: ${errorDetail}`);
			}
		} catch (error: any) {
			console.error("Error starting scan:", error);
			set({
				error: `Failed to start scan: ${error.message || "Network error or invalid response."}`,
				isLoading: false,
				scanRunning: false, // Ensure scanRunning is false if start fails
			});
		}
	},

	stopScan: async () => {
		set({ isLoading: true }); // Indicate loading while stopping
		get().disconnectWebSocket(); // Immediately close WS from client side

		try {
			const response = await fetch("http://localhost:8000/stop-scan");
			// Check status first, backend might return 400 if not running
			if (response.status === 400) {
				const errorResult: ErrorResponse = await response.json();
				throw new Error(
					errorResult.detail ||
						"Scan not running or already stopped.",
				);
			}
			if (!response.ok) {
				// Handle other non-400 errors
				throw new Error(
					`Server responded with status ${response.status}`,
				);
			}

			const result: SuccessResponse | ErrorResponse =
				await response.json();

			if (result.status === "success") {
				set({
					scanRunning: false,
					isLoading: false,
					activeTasks: 0,
				});
				console.log("Scan stop confirmed by server.");
			} else {
				// This case might not happen if non-ok responses are caught above, but good practice
				throw new Error(
					(result as ErrorResponse).message ||
						"Failed to stop scan (server error).",
				);
			}
		} catch (error: any) {
			console.error("Error stopping scan:", error);
			set({
				error: `Failed to stop scan: ${error.message || "Network error or server issue."}`,
				isLoading: false,
				// Optionally leave scanRunning true if stop fails? Or force false?
				// Let's assume stop failed, but user intended to stop, so set false.
				scanRunning: false,
			});
		}
	},

	checkScanStatus: async () => {
		// No loading state for background check, but clear errors
		// set({ error: null });
		try {
			const response = await fetch("http://localhost:8000/scan-status");
			if (!response.ok) {
				// Don't set a persistent error for a background check failure
				console.warn(
					`Scan status check failed: HTTP ${response.status}`,
				);
				// Assume not running if status check fails? Or retain current state?
				// Let's retain current state to avoid flickering UI on transient network issues.
				// set({ scanRunning: false, activeTasks: 0 });
				return;
			}
			const data: ScanStatusResponse = await response.json();
			set({
				scanRunning: data.scan_running,
				activeTasks: data.active_tasks,
				scanModel: data.selected_model,
			});

			// If status check reveals scan is running, ensure WebSocket is connected
			if (
				data.scan_running &&
				(!get().ws || get().ws?.readyState !== WebSocket.OPEN) &&
				!get().isConnecting
			) {
				console.log(
					"Status check found scan running, ensuring WebSocket connection...",
				);
				get().connectWebSocket();
			}
			// If status check reveals scan is NOT running, ensure WS is disconnected
			else if (!data.scan_running && get().ws) {
				console.log(
					"Status check found scan not running, ensuring WebSocket disconnection...",
				);
				get().disconnectWebSocket();
			}
		} catch (error: any) {
			console.error("Error checking scan status:", error);
			// Update UI with error message for better feedback
			set({
				error: `Failed to check scan status: ${error.message || "Network error or server issue."}`,
			});
		}
	},
}));
