import { Search } from "lucide-react";
import React from "react";
import { cn } from "../../utils/cn";

interface SearchInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
	value: string;
	onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
	placeholder?: string;
	className?: string;
	containerClassName?: string;
}

const SearchInput: React.FC<SearchInputProps> = ({
	value,
	onChange,
	placeholder = "Search...",
	className,
	containerClassName,
	...props
}) => {
	return (
		<div className={cn("relative flex-1", containerClassName)}>
			<Search
				className="absolute inset-y-0 top-1/2 left-2.5 my-auto -translate-y-1/2 transform text-gray-400/80"
				size={14}
				aria-hidden="true"
			/>
			<input
				type="text"
				placeholder={placeholder}
				value={value}
				onChange={onChange}
				className={cn(
					"w-full rounded-md border border-gray-700/80 bg-gray-800/90 py-1.5 pr-3 pl-8 text-xs placeholder-gray-400/90 shadow-sm focus:border-blue-600 focus:ring-1 focus:ring-blue-600 focus:outline-none",
					className,
				)}
				{...props}
			/>
		</div>
	);
};

export default SearchInput;
