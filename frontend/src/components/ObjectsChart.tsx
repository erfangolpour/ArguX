import { BarChart3 } from "lucide-react"; // Consistent icons
import React, { useMemo, useState } from "react";
import { useCameraStore } from "../store/cameraStore";
import { cn } from "../utils/cn";
import ChartContainer from "./common/ChartContainer"; // Import ChartContainer
import SearchInput from "./common/SearchInput"; // Import SearchInput
import SortControls from "./common/SortControls"; // Import SortControls

const ObjectsChart: React.FC = () => {
	// Use individual primitive selectors instead of object selectors
	const getObjectCounts = useCameraStore((state) => state.getObjectCounts);
	const selectedObjects = useCameraStore((state) => state.selectedObjects);
	const toggleObjectSelection = useCameraStore((state) => state.toggleObjectSelection);

	const objectCounts = getObjectCounts(); // Get sorted array

	const [searchTerm, setSearchTerm] = useState<string>("");
	const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
	const [sortField, setSortField] = useState<"count" | "alpha">("count");
	const [showAll, setShowAll] = useState<boolean>(false);

	// Filter and sort data based on search term and sort order
	const visibleObjects = useMemo(() => {
		let filtered = [...objectCounts];

		// Apply search filter (case-insensitive)
		if (searchTerm) {
			const term = searchTerm.toLowerCase();
			filtered = filtered.filter((item) =>
				item.name.toLowerCase().includes(term),
			);
		}

		// Sort the filtered data
		filtered.sort((a, b) => {
			if (sortField === "count") {
				return sortOrder === "asc"
					? a.count - b.count
					: b.count - a.count;
			} else {
				// Alphabetical sorting
				return sortOrder === "asc"
					? a.name.localeCompare(b.name)
					: b.name.localeCompare(a.name);
			}
		});

		// Limit the number of visible items if not showing all
		return showAll ? filtered : filtered.slice(0, 15); // Show top 15 by default
	}, [objectCounts, searchTerm, sortOrder, sortField, showAll]);

	// Calculate max count for bar scaling only based on *visible* objects
	const maxCount =
		visibleObjects.length > 0
			? Math.max(...visibleObjects.map((o) => o.count))
			: 0;

	// Function to get sort field label
	const getSortFieldLabel = (): string => {
		return sortField === "count" ? "Count" : "Label";
	};

	// Controls for the ChartContainer
	const chartControls = (
		<>
			<SearchInput
				value={searchTerm}
				onChange={(e) => setSearchTerm(e.target.value)}
				placeholder="Search objects..."
				containerClassName="flex-grow"
			/>
			<SortControls
				sortOrder={sortOrder}
				toggleSortOrder={() =>
					setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"))
				}
				sortFieldLabel={getSortFieldLabel()}
				toggleSortField={() =>
					setSortField((prev) =>
						prev === "count" ? "alpha" : "count",
					)
				}
				className="flex-shrink-0"
			/>
		</>
	);

	return (
		<ChartContainer
			title="Objects Detected"
			icon={
				<BarChart3
					className="mr-2 inline align-middle text-emerald-400"
					size={20}
				/>
			}
			controls={chartControls}
			infoText="Click item to filter cameras"
			isEmpty={objectCounts.length === 0}
			emptyText="No objects detected yet. Start a scan."
			selectedCount={selectedObjects.length}
		>
			{/* Container for list and show all button */}
			<div className="flex h-full flex-col">
				{/* Show All / Top 15 Button - Place above list */}
				{objectCounts.length > 15 && ( // Only show if there are more than 15 items total
					<div className="mb-2 flex-shrink-0 text-right">
						<button
							onClick={() => setShowAll(!showAll)}
							className="text-xs text-blue-400 hover:text-blue-300 hover:underline focus:outline-none"
						>
							{showAll
								? "Show Top 15"
								: `Show All (${objectCounts.length})`}
						</button>
					</div>
				)}

				{/* Scrollable List Area */}
				<div className="min-h-0 flex-1 overflow-y-auto pr-1 pb-2">
					<div className="space-y-1.5">
						{visibleObjects.map((object) => (
							<button // Use button for better accessibility
								key={object.name}
								className={cn(
									"group flex w-full cursor-pointer items-center rounded p-1 text-left transition-colors duration-100 ease-in-out focus:outline-none",
									selectedObjects.includes(object.name)
										? "bg-cyan-900/50 hover:bg-cyan-900/70 focus:bg-cyan-900/70"
										: "hover:bg-gray-800/60 focus:bg-gray-800/60",
								)}
								onClick={() =>
									toggleObjectSelection(object.name)
								}
							>
								{/* Label */}
								<div
									className="w-20 flex-shrink-0 truncate pr-2 text-xs text-gray-200 group-hover:text-white sm:w-24"
									title={object.name}
								>
									{object.name}
								</div>
								{/* Bar and Count */}
								<div className="flex flex-1 items-center">
									{/* Bar */}
									<div className="relative h-3 flex-1 overflow-hidden rounded-sm bg-gray-700/50">
										<div
											className="absolute top-0 left-0 h-full rounded-sm transition-[width]"
											style={{
												backgroundColor:
													selectedObjects.includes(
														object.name,
													)
														? "#22D3EE"
														: "#3B82F6", // Use selected/default colors
												width:
													maxCount > 0
														? `${(object.count / maxCount) * 100}%`
														: "0%",
											}}
										></div>
									</div>
									{/* Count */}
									<span className="ml-2 w-6 flex-shrink-0 text-right text-[11px] font-medium text-gray-400 group-hover:text-gray-200">
										{object.count}
									</span>
								</div>
							</button>
						))}
					</div>
				</div>
			</div>
		</ChartContainer>
	);
};

export default ObjectsChart;
