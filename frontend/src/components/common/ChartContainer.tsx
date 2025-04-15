import React from "react";
import { cn } from "../../utils/cn";
import Panel from "./Panel"; // Assuming Panel is in the same directory or adjust path

interface ChartContainerProps {
	title: string;
	icon: React.ReactNode;
	controls?: React.ReactNode; // Slot for search, sort, view mode buttons
	children: React.ReactNode; // The actual chart or list content
	infoText?: string; // Optional text like "Click to filter"
	isLoading?: boolean; // Optional loading state
	isEmpty?: boolean; // Optional empty state flag
	emptyText?: string; // Text for empty state
	selectedCount?: number; // Optional count of selected items for title badge
	className?: string; // Additional classes for the Panel
}

const ChartContainer: React.FC<ChartContainerProps> = ({
	title,
	icon,
	controls,
	children,
	infoText,
	isLoading = false,
	isEmpty = false,
	emptyText = "No data available yet",
	selectedCount,
	className,
}) => {
	return (
		<Panel className={cn("gap-2", className)}>
			{/* Header: Title and Selection Count */}
			<h2 className="flex items-center text-base font-semibold lg:text-lg">
				{icon}
				<span className="align-middle">{title}</span>
				{selectedCount !== undefined && selectedCount > 0 && (
					<span className="ml-2 align-middle text-xs font-medium text-blue-300">
						({selectedCount} selected)
					</span>
				)}
			</h2>

			{/* Controls Area */}
			{controls && (
				<div className="flex items-center space-x-2">{controls}</div>
			)}

			{/* Info Text Area */}
			{infoText && !isEmpty && !isLoading && (
				<div className="text-xs text-gray-400">{infoText}</div>
			)}

			{/* Content Area: Loading, Empty, or Chart/List */}
			<div className="flex flex-1 flex-col overflow-hidden">
				{isLoading ? (
					<div className="flex flex-1 items-center justify-center text-sm text-gray-400">
						{/* Add a spinner or loading indicator here */}
						Loading...
					</div>
				) : isEmpty ? (
					<div className="flex flex-1 items-center justify-center text-center text-sm text-gray-400">
						{emptyText}
					</div>
				) : (
					// Use min-h-0 on the direct child if it needs to scroll
					// The Panel already has overflow-hidden, so the child needs overflow-y-auto if needed
					<div className="min-h-0 flex-1">{children}</div>
				)}
			</div>
		</Panel>
	);
};

export default ChartContainer;
