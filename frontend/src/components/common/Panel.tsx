import React from "react";
import { cn } from "../../utils/cn";

interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
	children: React.ReactNode;
	className?: string;
}

const Panel: React.FC<PanelProps> = ({ children, className, ...props }) => {
	return (
		<div
			className={cn(
				"flex h-full flex-col gap-3 overflow-hidden rounded-lg border border-gray-800/60 bg-black/40 p-4 shadow-lg backdrop-blur-md",
				className, // Allow overriding/extending styles
			)}
			{...props}
		>
			{children}
		</div>
	);
};

export default Panel;
