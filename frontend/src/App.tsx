import { Radar, X } from "lucide-react";
import { useEffect } from "react";
import CameraDetailsPanel from "./components/CameraDetailsPanel";
import CountryChart from "./components/CountryChart";
import { Globe } from "./components/Globe/Globe";
import ObjectsChart from "./components/ObjectsChart";
import ScanPanel from "./components/ScanPanel/ScanPanel";
import Panel from "./components/common/Panel"; // Import the common Panel
import { useCameraStore } from "./store/cameraStore";
import { cn } from "./utils/cn"; // Import cn utility

function App() {
	const {
		scanRunning,
		selectedObjects,
		selectedCountries,
		filteredCameras,
		clearFilters,
		checkScanStatus,
		error, // Get error from store
		clearError, // Action to clear error
	} = useCameraStore();

	// Check scan status on component mount
	useEffect(() => {
		checkScanStatus();
	}, [checkScanStatus]);

	const hasFilters =
		selectedObjects.length > 0 || selectedCountries.length > 0;

	return (
		<div className="relative flex h-screen flex-col overflow-hidden ">
			{/* Globe Visualization (Center) */}
			<div className="relative h-full">
				{/* Filter Status Panel */}
				{hasFilters && (
					<Panel className="absolute inset-x-0 top-16 z-10 m-auto !h-auto w-fit max-w-[90%] flex-row flex-wrap items-center gap-x-3 px-4 py-2 text-xs sm:text-sm">
						{selectedObjects.length > 0 && (
							<div className="flex items-center">
								<span className="font-semibold text-blue-300">
									Objects:
								</span>
								<span className="ml-1.5 text-gray-200">
									{selectedObjects.join(", ")}
								</span>
							</div>
						)}
						{selectedObjects.length > 0 &&
							selectedCountries.length > 0 && (
								<span className="hidden text-gray-600 sm:inline">
									|
								</span>
							)}
						{selectedCountries.length > 0 && (
							<div className="flex items-center">
								<span className="font-semibold text-blue-300">
									Countries:
								</span>
								<span className="ml-1.5 text-gray-200">
									{selectedCountries.join(", ")}
								</span>
							</div>
						)}
						<span className="ml-2 font-medium text-green-300">
							({filteredCameras.length} match
							{filteredCameras.length === 1 ? "" : "es"})
						</span>
						<button
							className="ml-auto rounded-full p-0.5 text-red-400 transition-colors hover:bg-red-500/20 hover:text-red-300"
							onClick={clearFilters}
							title="Clear all filters"
						>
							<X size={16} />
							<span className="sr-only">Clear Filters</span>
						</button>
					</Panel>
				)}

				{/* Global Error Message Panel */}
				{error && (
					<Panel className="absolute inset-x-0 bottom-4 z-50 m-auto !h-auto w-fit max-w-[80%] flex-row items-center gap-4 !border-red-600/80 bg-red-900/60 px-4 py-2 text-xs sm:text-sm">
						<span className="font-medium text-red-100">
							{error}
						</span>
						<button
							className="ml-auto cursor-pointer rounded-full p-0.5 text-red-200 transition-colors hover:bg-red-500/30 hover:text-white"
							onClick={clearError}
							title="Dismiss error"
						>
							<X size={16} />
							<span className="sr-only">Dismiss Error</span>
						</button>
					</Panel>
				)}

				<div className="z-0 h-full">
					<Globe />
				</div>
			</div>

			{/* Dashboard Overlay */}
			<div className="pointer-events-none absolute inset-0 z-20 flex flex-col">
				{/* Header */}
				<header className="flex items-center justify-between px-4 py-3 sm:p-5">
					<div className="flex items-center">
						<Radar
							className="mr-2 text-blue-400"
							size={24}
							sm-size={28}
						/>
						<h1 className="bg-gradient-to-r from-blue-400 to-cyan-300 bg-clip-text text-xl font-bold text-transparent sm:text-2xl">
							ArguX
						</h1>
					</div>
					<div
						className={cn(
							"rounded-full px-3 py-1 text-xs sm:text-sm",
							scanRunning
								? "bg-green-500/20 text-green-300"
								: "bg-gray-700/30 text-gray-400",
						)}
					>
						{scanRunning ? "Scan in progress..." : "Idle"}
					</div>
				</header>

				{/* Main Content Area (Sidebars) */}
				<div className="flex grow items-stretch justify-between overflow-hidden p-2 sm:p-3">
					{/* Left Sidebar Container */}
					<div className="pointer-events-auto flex w-[45%] flex-col gap-2 sm:w-1/3 2xl:w-1/4">
						{/* Scan Panel takes fixed height */}
						<div className="flex-shrink-0">
							<ScanPanel />
						</div>
						{/* Objects chart takes remaining height */}
						<div className="flex-grow overflow-hidden">
							<ObjectsChart />
						</div>
					</div>

					{/* Right Sidebar Container */}
					<div className="pointer-events-auto flex w-[45%] flex-col gap-2 sm:w-1/3 2xl:w-1/4">
						{/* Camera Details takes half height */}
						<div className="h-1/2 overflow-hidden">
							<CameraDetailsPanel />
						</div>
						{/* Country Chart takes half height */}
						<div className="h-1/2 overflow-hidden">
							<CountryChart />
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}

export default App;
