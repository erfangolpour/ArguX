import React from "react";
import ReactCountryFlag from "react-country-flag";
import { Camera } from "../../types/types"; // Import Camera type

interface CameraTooltipProps {
	camera: Camera | null;
	visible: boolean;
	position: { x: number; y: number };
	displayBelow?: boolean; // Prop to control position relative to cursor
}

export const CameraTooltip: React.FC<CameraTooltipProps> = ({
	camera,
	visible,
	position,
	displayBelow = false, // Default to above cursor
}) => {
	if (!visible || !camera) return null;

	// Basic style for the tooltip container
	const style: React.CSSProperties = {
		left: `${position.x}px`,
		top: `${position.y}px`,
		transform: `translate(-50%, ${displayBelow ? "20px" : "calc(-100% - 20px)"})`, // Adjusted offset
		maxWidth: "280px", // Max width
		pointerEvents: "none", // Allow clicks to pass through
		transition: "opacity 0.2s ease-in-out", // Fade transition
		opacity: visible ? 1 : 0, // Control visibility via opacity
		zIndex: 1000, // Ensure it's on top
	};

	return (
		<div
			className="absolute rounded-md border border-blue-500/40 bg-gray-900/80 p-2.5 text-white shadow-xl backdrop-blur-md"
			style={style}
		>
			{/* Header: Location */}
			<div className="mb-1.5 flex items-center text-sm font-medium text-blue-300">
				<ReactCountryFlag
					countryCode={camera.geo?.countryCode || "XX"}
					svg
					style={{ width: "1em", height: "1em" }}
					className="mr-1.5 flex-shrink-0"
					title={camera.geo?.country || "Unknown Country"}
				/>
				<span className="truncate">
					{camera.geo?.city ? `${camera.geo.city}, ` : ""}
					{camera.geo?.country || "Unknown Location"}
				</span>
			</div>

			{/* Snapshot Preview */}
			{camera.snapshot && (
				<div className="mb-1.5 aspect-video w-full overflow-hidden rounded-sm bg-gray-700">
					<img
						src={camera.snapshot}
						alt="Camera snapshot"
						className="h-full w-full object-cover"
					/>
				</div>
			)}
			{!camera.snapshot && camera.error && (
				<div className="mb-1.5 flex aspect-video w-full items-center justify-center overflow-hidden rounded-sm bg-red-800/40 p-2 text-center text-xs text-red-200">
					<span>{camera.error || "Snapshot unavailable"}</span>
				</div>
			)}

			{/* Details: IP and Detections */}
			<div className="space-y-1 text-xs">
				{camera.geo?.query && (
					<div className="text-gray-300">
						IP:{" "}
						<span className="font-mono">{camera.geo.query}</span>
					</div>
				)}
				<div>
					<span className="text-gray-400">
						Detected ({camera.detections.length}):
					</span>
					{camera.detections.length > 0 ? (
						<div className="mt-1 flex flex-wrap gap-1">
							{camera.detections.slice(0, 5).map(
								(
									detection,
									idx, // Limit displayed detections
								) => (
									<span
										key={idx}
										className="rounded-sm bg-blue-600/50 px-1 py-0.5 text-[10px] text-blue-100"
									>
										{detection.label} (
										{Math.round(detection.confidence * 100)}
										%)
									</span>
								),
							)}
							{camera.detections.length > 5 && (
								<span className="rounded-sm bg-gray-600/50 px-1 py-0.5 text-[10px] text-gray-300">
									+{camera.detections.length - 5} more
								</span>
							)}
						</div>
					) : (
						<span className="ml-1 text-gray-400/80">
							No detections
						</span>
					)}
				</div>
				<div className="pt-1 text-[10px] text-gray-500 opacity-80">
					Updated: {camera.timestamp.toLocaleTimeString()}{" "}
					{/* Show time only for brevity */}
				</div>
			</div>
		</div>
	);
};
