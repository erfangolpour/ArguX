import { SortAsc, SortDesc } from "lucide-react";
import { cn } from "../../utils/cn";

interface SortControlsProps<T extends string> {
	sortOrder: "asc" | "desc";
	toggleSortOrder: () => void;
	sortField?: T; // Optional: If sorting by different fields
	toggleSortField?: () => void; // Optional: If sorting by different fields
	sortFieldLabel?: string; // Optional: Display label for sort field
	className?: string;
}

const SortControls = <T extends string>({
	sortOrder,
	toggleSortOrder,
	toggleSortField,
	sortFieldLabel,
	className,
}: SortControlsProps<T>) => {
	return (
		<div className={cn("flex items-center gap-1.5", className)}>
			{/* Sort Order Button */}
			<button
				onClick={toggleSortOrder}
				className="cursor-pointer rounded-md border border-gray-700/80 bg-gray-800/90 p-1.5 text-gray-300 shadow-sm transition-colors hover:bg-gray-700/90"
				title={
					sortOrder === "desc"
						? "Sorted descending"
						: "Sorted ascending"
				}
			>
				{sortOrder === "desc" ? (
					<SortDesc size={14} />
				) : (
					<SortAsc size={14} />
				)}
				<span className="sr-only">Toggle sort order</span>
			</button>

			{/* Sort Field Button (Optional) */}
			{toggleSortField && sortFieldLabel && (
				<button
					onClick={toggleSortField}
					className="cursor-pointer rounded-md border border-gray-700/80 bg-gray-800/90 px-2 py-1.5 text-xs text-gray-300 shadow-sm transition-colors hover:bg-gray-700/90"
					title={`Sort by ${sortFieldLabel}`}
				>
					{sortFieldLabel}
				</button>
			)}
		</div>
	);
};

export default SortControls;
