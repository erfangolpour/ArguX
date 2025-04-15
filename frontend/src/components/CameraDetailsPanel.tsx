import {
	ArrowLeft,
	Building,
	ExternalLink,
	Eye,
	Info,
	MapPin,
	Monitor,
	Network,
} from "lucide-react";
import React, { useMemo, useState } from "react";
import ReactCountryFlag from "react-country-flag";
import { useCameraStore } from "../store/cameraStore";
import { Camera as CameraType } from "../types/types"; // Use Camera type
import { cn } from "../utils/cn"; // Import cn
import Panel from "./common/Panel";
import SearchInput from "./common/SearchInput"; // Import SearchInput
import SortControls from "./common/SortControls"; // Import SortControls

// Placeholder for missing snapshot
const placeholderImage =
	'data:image/svg+xml;charset=UTF-8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 150"><rect width="100%" height="100%" fill="%23374151"/></svg>';

const CameraDetailsPanel: React.FC = () => {
	// Use individual primitive selectors with filteredCameras instead of cameras
	const filteredCameras = useCameraStore((state) => state.filteredCameras);
	const selectedCamera = useCameraStore((state) => state.selectedCamera);
	const setSelectedCamera = useCameraStore(
		(state) => state.setSelectedCamera,
	);
	const searchTerm = useCameraStore((state) => state.searchTerm);
	const setSearchTerm = useCameraStore((state) => state.setSearchTerm);
	const scanRunning = useCameraStore((state) => state.scanRunning);
	const cameraTypeFilter = useCameraStore((state) => state.cameraTypeFilter);
	const setCameraTypeFilter = useCameraStore(
		(state) => state.setCameraTypeFilter,
	);

	const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
	type SortField =
		| keyof CameraType
		| `geo.${keyof CameraType["geo"]}`
		| "detectionsCount";
	const [sortField, setSortField] = useState<SortField>("timestamp");

	// Sort cameras based on the filtered list instead of applying additional filters
	const sortedCameras = useMemo(() => {
		let sorted = [...filteredCameras]; // Now sorting the already filtered list

		// Just sort based on selected field and order - no more filtering here
		sorted.sort((a, b) => {
			let valA: any, valB: any;

			switch (sortField) {
				case "timestamp":
					valA = a.timestamp.getTime();
					valB = b.timestamp.getTime();
					break;
				case "detectionsCount":
					valA = a.detections.length;
					valB = b.detections.length;
					break;
				case "geo.country":
					valA = a.geo.country || "";
					valB = b.geo.country || "";
					return sortOrder === "asc"
						? valA.localeCompare(valB)
						: valB.localeCompare(valA);
				case "geo.city":
					valA = a.geo.city || "";
					valB = b.geo.city || "";
					return sortOrder === "asc"
						? valA.localeCompare(valB)
						: valB.localeCompare(valA);
				default:
					valA = 0;
					valB = 0;
			}

			return sortOrder === "asc"
				? valA < valB
					? -1
					: 1
				: valB < valA
					? -1
					: 1;
		});

		return sorted;
	}, [filteredCameras, sortOrder, sortField]);

	// Toggle sort field - cycle through options
	const cycleSortField = () => {
		const fields: SortField[] = [
			"timestamp",
			"detectionsCount",
			"geo.country",
			"geo.city",
		];
		const currentIndex = fields.indexOf(sortField);
		const nextIndex = (currentIndex + 1) % fields.length;
		setSortField(fields[nextIndex]);
	};

	const getSortFieldLabel = (): string => {
		switch (sortField) {
			case "timestamp":
				return sortOrder === "desc" ? "Latest" : "Oldest";
			case "detectionsCount":
				return "Objects";
			case "geo.country":
				return "Country";
			case "geo.city":
				return "City";
			default:
				return "Timestamp";
		}
	};

	// Camera Type Filter Controls
	const CameraTypeFilterControls: React.FC = () => {
		return (
			<div className="flex items-center space-x-1 text-xs">
				{/* <Filter size={12} className="mr-2 text-gray-400" /> */}
				<button
					onClick={() => setCameraTypeFilter("all")}
					className={cn(
						"cursor-pointer rounded px-1.5 py-0.5 transition-colors",
						cameraTypeFilter === "all"
							? "bg-blue-900/60 text-blue-100"
							: "text-gray-400 hover:bg-gray-700 hover:text-gray-200",
					)}
				>
					All
				</button>
				<button
					onClick={() => setCameraTypeFilter("errors")}
					className={cn(
						"cursor-pointer rounded px-1.5 py-0.5 transition-colors",
						cameraTypeFilter === "errors"
							? "bg-red-900/60 text-red-100"
							: "text-gray-400 hover:bg-gray-700 hover:text-gray-200",
					)}
				>
					Errors
				</button>
				<button
					onClick={() => setCameraTypeFilter("empty")}
					className={cn(
						"cursor-pointer rounded px-1.5 py-0.5 transition-colors",
						cameraTypeFilter === "empty"
							? "bg-yellow-900/60 text-yellow-100"
							: "text-gray-400 hover:bg-gray-700 hover:text-gray-200",
					)}
				>
					Empty
				</button>
				<button
					onClick={() => setCameraTypeFilter("detected")}
					className={cn(
						"cursor-pointer rounded px-1.5 py-0.5 transition-colors",
						cameraTypeFilter === "detected"
							? "bg-green-900/60 text-green-100"
							: "text-gray-400 hover:bg-gray-700 hover:text-gray-200",
					)}
				>
					Detections
				</button>
			</div>
		);
	};

	// --- Render Logic ---

	// 1. Selected Camera View
	if (selectedCamera) {
		return (
			<Panel>
				{/* Header */}
				<div className="flex flex-shrink-0 items-center">
					<button
						onClick={() => setSelectedCamera(null)}
						className="mr-1.5 -ml-1.5 rounded-full p-1 text-gray-300 transition-colors hover:bg-gray-700/80 hover:text-white"
						title="Back to camera list"
					>
						<ArrowLeft className="align-middle" size={16} />
						<span className="sr-only">Back</span>
					</button>
					<Monitor
						className="mr-2 inline align-middle text-blue-400"
						size={18}
					/>
					<h2 className="truncate align-middle text-base font-semibold lg:text-lg">
						Camera Details
					</h2>
				</div>

				{/* Scrollable Content Area */}
				<div className="flex-1 space-y-3 overflow-y-auto pr-1 pb-2">
					{/* Snapshot with timestamp */}
					<div className="relative aspect-video w-full overflow-hidden rounded-md bg-gray-700">
						<img
							src={selectedCamera.snapshot || placeholderImage}
							alt={`Snapshot from ${selectedCamera.geo.city || "Unknown"}, ${selectedCamera.geo.country || "Unknown"}`}
							className="h-full w-full object-cover"
							onError={(e) => {
								const target = e.target as HTMLImageElement;
								target.onerror = null; // Prevent infinite loop
								target.src = placeholderImage;
							}}
						/>
						<div className="absolute bottom-1.5 left-1.5 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white backdrop-blur-sm">
							{selectedCamera.timestamp.toLocaleString([], {
								year: "numeric",
								month: "short",
								day: "numeric",
								hour: "2-digit",
								minute: "2-digit",
								second: "2-digit",
							})}
						</div>
						{/* Error overlay */}
						{selectedCamera.error && !selectedCamera.processed && (
							<div className="absolute inset-0 flex items-center justify-center bg-red-900/70 p-2 text-center">
								<span className="text-xs text-red-100">
									{selectedCamera.error}
								</span>
							</div>
						)}
					</div>

					{/* Details Section */}
					<div className="space-y-2 text-xs">
						<div>
							<h3 className="mb-0.5 flex items-center text-[11px] font-medium text-gray-400">
								<Network size={11} className="mr-1" /> IP
								Address / URL
							</h3>
							<p className="font-mono break-all">
								{selectedCamera.geo.query || selectedCamera.url}
							</p>
							<a
								href={selectedCamera.url}
								target="_blank"
								rel="noopener noreferrer"
								className="mt-1 inline-flex items-center text-blue-400 hover:text-blue-300 hover:underline"
							>
								<ExternalLink size={11} className="mr-1" />
								Open Live Stream
							</a>
						</div>

						<div>
							<h3 className="mb-0.5 flex items-center text-[11px] font-medium text-gray-400">
								<MapPin size={11} className="mr-1" /> Location
							</h3>
							<div className="flex items-center">
								<ReactCountryFlag
									countryCode={
										selectedCamera.geo.countryCode || "XX"
									}
									svg
									style={{ width: "1em", height: "1em" }} // Ensure consistent size
									className="mr-1.5 flex-shrink-0"
									title={
										selectedCamera.geo.country ||
										"Unknown Country"
									}
								/>
								<span className="truncate">
									{selectedCamera.geo.city
										? `${selectedCamera.geo.city}, `
										: ""}
									{selectedCamera.geo.regionName
										? `${selectedCamera.geo.regionName}, `
										: ""}
									{selectedCamera.geo.country ||
										"Unknown Location"}
								</span>
							</div>
							{selectedCamera.geo.lat != null &&
								selectedCamera.geo.lon != null && (
									<p className="mt-0.5 text-gray-400/80">
										{selectedCamera.geo.lat.toFixed(4)},{" "}
										{selectedCamera.geo.lon.toFixed(4)}
									</p>
								)}
						</div>

						{selectedCamera.geo.as_info && (
							<div>
								<h3 className="mb-0.5 flex items-center text-[11px] font-medium text-gray-400">
									<Building size={11} className="mr-1" />{" "}
									Network / Org
								</h3>
								<p>
									{selectedCamera.geo.as_info}
									{selectedCamera.geo.org
										? ` (${selectedCamera.geo.org})`
										: ""}
								</p>
							</div>
						)}

						{selectedCamera.detections.length > 0 && (
							<div>
								<h3 className="mb-0.5 flex items-center text-[11px] font-medium text-gray-400">
									<Eye size={11} className="mr-1" /> Detected
									Objects ({selectedCamera.detections.length})
								</h3>
								<div className="mt-1 flex flex-wrap gap-1">
									{selectedCamera.detections.map(
										(detection, index) => (
											<div
												key={index}
												className="flex items-center rounded bg-blue-900/60 px-1.5 py-0.5"
											>
												<span className="mr-1">
													{detection.label}
												</span>
												<span className="text-[10px] text-blue-300 opacity-80">
													(
													{Math.round(
														detection.confidence *
															100,
													)}
													%)
												</span>
											</div>
										),
									)}
								</div>
							</div>
						)}
						{selectedCamera.detections.length === 0 &&
							selectedCamera.processed && (
								<div>
									<h3 className="mb-0.5 flex items-center text-[11px] font-medium text-gray-400">
										<Eye size={11} className="mr-1" />{" "}
										Detected Objects
									</h3>
									<p className="text-gray-400/80">
										No objects detected.
									</p>
								</div>
							)}
					</div>
				</div>
			</Panel>
		);
	}

	// 2. Camera List View
	return (
		<Panel>
			{/* Header */}
			<div className="flex flex-shrink-0 items-center justify-between">
				<h2 className="flex items-center text-base font-semibold lg:text-lg">
					<Monitor
						className="mr-2 inline align-middle text-blue-400"
						size={18}
					/>
					<span className="align-middle">Cameras Found</span>
					<span className="ml-2 text-sm font-normal text-gray-400">
						({filteredCameras.length})
					</span>
				</h2>
			</div>

			{/* Search and sort controls */}
			<CameraTypeFilterControls />
			<div className="flex flex-shrink-0 flex-wrap items-center gap-2">
				<div className="flex-1">
					<SearchInput
						value={searchTerm}
						onChange={(e) => setSearchTerm(e.target.value)}
						placeholder="Search IP, location, object..."
					/>
				</div>
				<div className="flex items-center">
					<SortControls
						sortOrder={sortOrder}
						toggleSortOrder={() =>
							setSortOrder((prev) =>
								prev === "asc" ? "desc" : "asc",
							)
						}
						toggleSortField={cycleSortField}
						sortFieldLabel={getSortFieldLabel()}
					/>
				</div>
			</div>

			{/* Camera List Area */}
			<div className="min-h-0 flex-1 overflow-y-auto pr-1 pb-1">
				{filteredCameras.length === 0 ? (
					// Placeholder when no cameras loaded yet
					<div className="flex h-full flex-col items-center justify-center text-center text-sm text-gray-400">
						<Info size={32} className="mb-3 text-gray-500" />
						<p>
							{scanRunning
								? "Scanning for cameras..."
								: searchTerm || cameraTypeFilter !== "all"
									? "No cameras match your search or filters"
									: "Start a scan to find cameras"}
						</p>
						<p className="text-xs text-gray-500">
							{scanRunning
								? "Results will appear here."
								: searchTerm || cameraTypeFilter !== "all"
									? "Try adjusting your search terms or filters."
									: "Use the Scan Control panel."}
						</p>
					</div>
				) : (
					// Actual camera list
					<div className="grid grid-cols-1 gap-1.5">
						{sortedCameras.map((cam) => (
							<button // Use button for accessibility and click handling
								key={cam.id}
								className={cn(
									"group w-full cursor-pointer overflow-hidden rounded-md text-left transition-colors duration-100 ease-in-out hover:bg-gray-700/60 focus:bg-gray-700/60 focus:outline-none",
									cam.error && !cam.processed
										? "border border-red-700/50 hover:bg-red-900/20"
										: "border border-transparent", // Highlight errors
								)}
								onClick={() => setSelectedCamera(cam)}
							>
								<div className="flex">
									{/* Thumbnail */}
									<div className="relative w-20 flex-shrink-0 bg-gray-800 group-hover:opacity-90">
										<img
											src={
												cam.snapshot || placeholderImage
											}
											alt={`Camera in ${cam.geo.city || "Unknown"}`}
											className="h-full w-full object-cover"
											loading="lazy" // Lazy load images
											onError={(e) => {
												const target =
													e.target as HTMLImageElement;
												target.onerror = null;
												target.src = placeholderImage;
											}}
										/>
										{/* Error indicator on thumbnail */}
										{cam.error && !cam.processed && (
											<div className="absolute inset-0 flex items-center justify-center bg-red-900/60">
												<Info
													size={16}
													className="text-red-200"
												/>
											</div>
										)}
									</div>

									{/* Camera info */}
									<div className="flex-1 overflow-hidden p-1.5 pl-2 text-xs">
										<div className="flex items-start justify-between">
											<div className="truncate font-mono text-[11px] font-medium text-gray-200 group-hover:text-white">
												{cam.geo.query || cam.url}
											</div>
											<div className="ml-1 flex-shrink-0">
												<ReactCountryFlag
													countryCode={
														cam.geo.countryCode ||
														"XX"
													}
													svg
													style={{
														width: "1em",
														height: "1em",
													}}
													title={
														cam.geo.country ||
														"Unknown Country"
													}
												/>
											</div>
										</div>
										<div className="mt-0.5 truncate text-gray-400 group-hover:text-gray-300">
											{cam.geo.city
												? `${cam.geo.city}, `
												: ""}
											{cam.geo.country || "Unknown"}
										</div>
										<div className="mt-0.5 truncate text-gray-500 group-hover:text-gray-400">
											{cam.error && !cam.processed ? (
												<span className="text-red-400">
													{cam.error}
												</span>
											) : cam.detections.length > 0 ? (
												cam.detections
													.slice(0, 3)
													.map((det) => det.label)
													.join(", ") +
												(cam.detections.length > 3
													? "..."
													: "")
											) : cam.processed ? (
												"No objects detected"
											) : (
												"Processing..."
											)}
										</div>
									</div>
								</div>
							</button>
						))}
					</div>
				)}
			</div>
		</Panel>
	);
};

export default CameraDetailsPanel;
