// frontend/src/components/CountryChart.tsx
import { Globe, ListFilter, ListTree } from "lucide-react"; // Updated icons
import React, { useMemo, useState } from "react";
import ReactCountryFlag from "react-country-flag";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { useCameraStore } from "../store/cameraStore";
import { CountryCount } from "../types/types";
import { cn } from "../utils/cn";
import ChartContainer from "./common/ChartContainer"; // Import ChartContainer
import SearchInput from "./common/SearchInput"; // Import SearchInput
import SortControls from "./common/SortControls"; // Import SortControls

// More distinct colors, ensuring selected color isn't in the list
const COLORS = [
	"#3B82F6", // blue-500
	"#10B981", // emerald-500
	"#F59E0B", // amber-500
	"#EF4444", // red-500
	"#8B5CF6", // violet-500
	"#0EA5E9", // sky-500
	"#14B8A6", // teal-500
	"#F97316", // orange-500
	"#EC4899", // pink-500
	"#6366F1", // indigo-500
];
const SELECTED_COLOR = "#22D3EE"; // cyan-400 (Distinct color)

// Intl display names for full country names
let regionNames = new Intl.DisplayNames(["en"], { type: "region" });

const CountryChart: React.FC = () => {
	// Use individual primitive selectors instead of object selector pattern
	const getCountryCounts = useCameraStore((state) => state.getCountryCounts);
	const selectedCountries = useCameraStore((state) => state.selectedCountries);
	const toggleCountrySelection = useCameraStore((state) => state.toggleCountrySelection);

	const countryCounts = getCountryCounts(); // Get the sorted array

	const [searchTerm, setSearchTerm] = useState<string>("");
	const [sortOrder, setSortOrder] = useState<"asc" | "desc">("desc");
	const [viewMode, setViewMode] = useState<"pie" | "list">("pie");

	// Filter and sort data based on search term and sort order
	const visibleCountries = useMemo(() => {
		let filtered = [...countryCounts];

		// Apply search filter (case-insensitive)
		if (searchTerm) {
			const term = searchTerm.toLowerCase();
			filtered = filtered.filter((item) => {
				// Handle potential errors if item.name is not a valid country code
				try {
					const countryName = item.name
						? regionNames.of(item.name)
						: null;
					return (
						(countryName &&
							countryName.toLowerCase().includes(term)) ||
						item.name.toLowerCase().includes(term)
					);
				} catch (e) {
					console.warn(
						`Invalid country code encountered: ${item.name}`,
					);
					return false; // Exclude invalid codes from search results
				}
			});
		}

		// Re-sort based on current sortOrder (original data is already sorted desc by default)
		if (sortOrder === "asc") {
			filtered.sort((a, b) => a.count - b.count);
		}
		// No need for else, as original is already desc

		// Limit for pie chart view
		return viewMode === "pie" ? filtered.slice(0, 10) : filtered;
	}, [countryCounts, searchTerm, sortOrder, viewMode]);

	// Handle pie sector click
	const handlePieClick = (data: any, _index: number) => {
		if (data && data.name) {
			toggleCountrySelection(data.name); // Pass the country code (name)
		}
	};

	// Get color for a country based on selection state
	const getCountryColor = (entry: CountryCount, index: number): string => {
		return selectedCountries.includes(entry.name)
			? SELECTED_COLOR
			: COLORS[index % COLORS.length];
	};

	// Custom tooltip for Pie chart
	const CustomTooltip = ({ active, payload }: any) => {
		if (active && payload && payload.length) {
			const countryCode = payload[0].payload.name; // Access name from payload.payload
			let countryName = countryCode;
			try {
				countryName = regionNames.of(countryCode) || countryCode;
			} catch {
				/* Ignore errors for invalid codes */
			}

			return (
				<div className="rounded border border-gray-700 bg-gray-800/90 px-2.5 py-1.5 text-xs text-white shadow-md backdrop-blur-sm">
					<div className="mb-0.5 flex items-center font-medium">
						<ReactCountryFlag
							countryCode={countryCode}
							svg
							style={{ width: "1em", height: "1em" }}
							className="mr-1.5 flex-shrink-0"
						/>
						<span className="align-middle">{countryName}</span>
					</div>
					Count:{" "}
					<span className="font-semibold">{payload[0].value}</span>
				</div>
			);
		}
		return null;
	};

	// Controls for the ChartContainer
	const chartControls = (
		<>
			<SearchInput
				value={searchTerm}
				onChange={(e) => setSearchTerm(e.target.value)}
				placeholder="Search countries..."
				containerClassName="flex-grow"
			/>
			<SortControls
				sortOrder={sortOrder}
				toggleSortOrder={() =>
					setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"))
				}
				className="flex-shrink-0"
			/>
			<button
				onClick={() =>
					setViewMode((prev) => (prev === "pie" ? "list" : "pie"))
				}
				className="rounded-md border border-gray-700/80 bg-gray-800/90 p-1.5 text-gray-300 shadow-sm transition-colors hover:bg-gray-700/90"
				title={
					viewMode === "pie"
						? "Switch to list view"
						: "Switch to pie chart view"
				}
			>
				{viewMode === "pie" ? (
					<ListTree size={14} />
				) : (
					<ListFilter size={14} />
				)}
				<span className="sr-only">Toggle View Mode</span>
			</button>
		</>
	);

	return (
		<ChartContainer
			title="Country Distribution"
			icon={
				<Globe
					className="mr-2 inline align-middle text-blue-400"
					size={20}
				/>
			}
			controls={chartControls}
			infoText="Click item to filter cameras"
			isEmpty={countryCounts.length === 0}
			emptyText="No country data yet. Start a scan."
			selectedCount={selectedCountries.length}
		>
			{viewMode === "pie" ? (
				// Pie Chart View (ensure it takes full height of container)
				<ResponsiveContainer width="100%" height="100%">
					<PieChart>
						<Pie
							data={visibleCountries}
							cx="50%"
							cy="50%"
							labelLine={false}
							outerRadius="80%" // Adjust radius
							innerRadius="40%" // Make donut
							fill="#8884d8"
							dataKey="count"
							nameKey="name" // Use country code as key
							label={(
								{ name, percent }, // Custom label
							) => `${name} ${(percent * 100).toFixed(0)}%`}
							onClick={handlePieClick} // Use correct handler signature
							cursor="pointer"
							stroke="rgba(0,0,0,0.2)" // Add subtle stroke
							strokeWidth={1}
							fontSize={10}
						>
							{visibleCountries.map((entry, index) => (
								<Cell
									key={`cell-${entry.name}-${index}`}
									fill={getCountryColor(entry, index)}
								/>
							))}
						</Pie>
						<Tooltip content={<CustomTooltip />} />
					</PieChart>
				</ResponsiveContainer>
			) : (
				// List View (make it scrollable)
				<div className="h-full overflow-y-auto pr-1 pb-2">
					<table className="w-full text-xs">
						<thead className="sticky top-0 z-10 bg-black/40 backdrop-blur-md">
							<tr>
								<th className="w-3/5 py-1.5 pl-1 text-left font-medium text-gray-400">
									Country
								</th>
								<th className="w-2/5 py-1.5 pr-1 text-right font-medium text-gray-400">
									Count
								</th>
							</tr>
						</thead>
						<tbody className="divide-y divide-gray-700/50">
							{visibleCountries.map((country) => {
								let countryName = country.name;
								try {
									countryName =
										regionNames.of(country.name) ||
										country.name;
								} catch {
									/* ignore */
								}

								return (
									<tr
										key={country.name}
										className={cn(
											"cursor-pointer transition-colors duration-100 ease-in-out hover:bg-gray-700/60",
											selectedCountries.includes(
												country.name,
											)
												? "bg-blue-900/40 hover:bg-blue-900/60"
												: "hover:bg-gray-800/60",
										)}
										onClick={() =>
											toggleCountrySelection(country.name)
										}
									>
										<td
											className="truncate py-1.5 pl-1"
											title={countryName}
										>
											<ReactCountryFlag
												countryCode={country.name}
												svg
												style={{
													width: "1em",
													height: "1em",
												}}
												className="mr-1.5 inline align-middle"
											/>
											<span className="align-middle">
												{countryName}
											</span>
										</td>
										<td className="py-1.5 pr-1 text-right font-medium">
											{country.count}
										</td>
									</tr>
								);
							})}
						</tbody>
					</table>
				</div>
			)}
		</ChartContainer>
	);
};

export default CountryChart;
