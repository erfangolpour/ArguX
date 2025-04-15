// Matches backend Pydantic model GeoLocation
export interface Geo {
	country?: string | null;
	countryCode?: string | null;
	region?: string | null; // Added region for completeness if available
	regionName?: string | null;
	city?: string | null;
	zip?: string | null; // Added zip
	lat?: number | null; // latitude
	lon?: number | null; // longitude
	timezone?: string | null;
	isp?: string | null; // Internet Service Provider
	org?: string | null; // organization
	as_info?: string | null; // AS number and name (matches alias 'as')
	query?: string | null; // IP address
}

// Matches backend Pydantic model Detection
export interface Detection {
	label: string;
	confidence: number;
}

// API Response Types
export interface SuccessResponse<T = any> {
	status: "success";
	message?: string;
	data?: T;
}

export interface ErrorResponse {
	status: "error";
	message: string;
	detail?: string;
}

export interface ScanStatusResponse {
	scan_running: boolean;
	active_tasks: number;
	selected_model: string;
}

// Backend result structure matching CameraScanResult Pydantic model
export interface CameraScanResult {
	id: string;
	url: string;
	geo: Geo | null;
	detections: Detection[];
	snapshot: string | null;
	timestamp: string; // Comes as ISO string from backend
	processed: boolean;
	error?: string | null;
}

// Frontend representation, derived from CameraScanResult
// Includes parsed Date and guaranteed Geo object
export interface Camera {
	id: string; // Celery Task ID
	url: string; // Should be valid HttpUrl
	geo: Geo; // Always defined, uses defaults if backend data is missing
	detections: Detection[];
	snapshot: string | null; // Base64 encoded image string or null
	timestamp: Date; // Parsed timestamp
	processed: boolean; // Was processing successful?
	error?: string | null; // Processing error message
}

// Alias for Camera to keep naming consistent with store
export type AppCamera = Camera;

// For UI display (Charts)
export interface ObjectCount {
	name: string;
	count: number;
}

export interface CountryCount {
	name: string; // Country Code (e.g., 'US')
	count: number;
}
