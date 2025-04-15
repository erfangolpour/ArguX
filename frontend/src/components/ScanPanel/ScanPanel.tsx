import {
	Globe,
	Loader2,
	ScanLine,
	SquareTerminal,
	StopCircle,
} from "lucide-react";
import React, { useCallback, useEffect, useState } from "react"; // Added useCallback
import ReactCountryFlag from "react-country-flag";
import Select, { FormatOptionLabelMeta } from "react-select";
import { useCameraStore } from "../../store/cameraStore";
import Panel from "../common/Panel";
import { countryOptions, modelOptions } from "./scanOptions";

// Custom format function for react-select options
const formatOptionLabel = (
	{ value, name, label }: { value: string; name?: string; label?: string },
	formatOptionLabelMeta: FormatOptionLabelMeta<{
		value: string;
		name?: string;
		label?: string;
	}>,
): React.ReactNode => {
	if (formatOptionLabelMeta.context === "menu") {
		// Menu item rendering
		if (name) {
			// Country option
			return (
				<div className="flex items-center">
					<ReactCountryFlag
						countryCode={value}
						svg
						style={{ width: "1em", height: "1em" }}
						className="mr-2 flex-shrink-0"
					/>
					<span className="align-middle">{name}</span>
				</div>
			);
		} else if (label) {
			// Model option
			return <div className="flex items-center">{label}</div>;
		}
	}
	// Value display rendering (when an option is selected)
	if (name) {
		// Selected country
		return (
			<div className="flex items-center text-xs">
				<ReactCountryFlag
					countryCode={value}
					svg
					style={{ width: "1em", height: "1em" }}
					className="mr-1.5"
				/>
				<span className="align-middle">{name}</span>
			</div>
		);
	} else if (label) {
		// Selected model
		return <div className="text-xs">{label}</div>;
	}
	return value; // Fallback
};

const ScanPanel: React.FC = () => {
	// Use individual selectors instead of a combined object selector
	const startScan = useCameraStore((state) => state.startScan);
	const stopScan = useCameraStore((state) => state.stopScan);
	const scanRunning = useCameraStore((state) => state.scanRunning);
	const isStoreLoading = useCameraStore((state) => state.isLoading);
	const scanModel = useCameraStore((state) => state.scanModel);

	// Local state for selections before starting scan
	const [selectedCountry, setSelectedCountry] = useState<{
		value: string;
		name: string;
	} | null>(null);

	// Initialize with a default value first
	const [selectedModel, setSelectedModel] = useState<{
		value: string;
		label: string;
	}>(modelOptions[0]);

	// Use callback to memoize the model option finder function
	const findModelOption = useCallback((modelName: string) => {
		return (
			modelOptions.find((opt) => opt.value === modelName) ||
			modelOptions[0]
		);
	}, []);

	// Update local state when the store value changes
	useEffect(() => {
		const modelOption = findModelOption(scanModel);
		setSelectedModel(modelOption);
	}, [scanModel, findModelOption]);

	const handleStartScan = async () => {
		// Pass optional country value and mandatory model value
		await startScan(selectedCountry?.value, selectedModel?.value);
	};

	const handleStopScan = async () => {
		await stopScan();
	};

	// Styles for react-select
	const selectStyles = {
		menuPortal: (base: any) => ({ ...base, zIndex: 9999 }),
		control: (base: any, state: any) => ({
			...base,
			backgroundColor: "rgba(31, 41, 55, 0.8)", // bg-gray-800 with opacity
			borderColor: state.isFocused
				? "rgb(59, 130, 246)"
				: "rgb(55, 65, 81)", // border-gray-700 / focus:border-blue-500
			boxShadow: state.isFocused ? "0 0 0 1px rgb(59, 130, 246)" : "none", // focus:ring-1 focus:ring-blue-500
			"&:hover": {
				borderColor: state.isFocused
					? "rgb(59, 130, 246)"
					: "rgb(75, 85, 99)", // hover:border-gray-600
			},
			minHeight: "32px", // Smaller height
			height: "32px",
			fontSize: "0.75rem", // text-xs
		}),
		valueContainer: (base: any) => ({
			...base,
			padding: "0 8px", // Adjust padding
			height: "32px",
		}),
		input: (base: any) => ({ ...base, color: "white", margin: "0px" }),
		indicatorSeparator: () => ({ display: "none" }), // Hide separator
		dropdownIndicator: (base: any) => ({
			...base,
			padding: "4px",
			color: "rgb(156, 163, 175)",
		}), // Adjust padding & color
		clearIndicator: (base: any) => ({
			...base,
			padding: "4px",
			color: "rgb(156, 163, 175)",
		}),
		menu: (base: any) => ({
			...base,
			backgroundColor: "rgb(31, 41, 55)", // bg-gray-800
			borderColor: "rgb(55, 65, 81)", // border-gray-700
			borderWidth: "1px",
			borderRadius: "0.375rem", // rounded-md
		}),
		option: (base: any, state: any) => ({
			...base,
			backgroundColor: state.isSelected
				? "rgb(37, 99, 235)"
				: state.isFocused
					? "rgb(55, 65, 81)"
					: "transparent", // bg-blue-700 (selected), bg-gray-700 (focused)
			color: state.isSelected ? "white" : "rgb(209, 213, 219)", // text-gray-200
			"&:active": {
				backgroundColor: state.isSelected
					? "rgb(37, 99, 235)"
					: "rgb(75, 85, 99)", // active:bg-gray-600
			},
			fontSize: "0.75rem", // text-xs
			padding: "6px 12px", // Adjust padding
		}),
		singleValue: (base: any) => ({ ...base, color: "white" }),
		placeholder: (base: any) => ({ ...base, color: "rgb(156, 163, 175)" }), // text-gray-400
	};

	return (
		<Panel>
			<h2 className="flex flex-shrink-0 items-center text-base font-semibold lg:text-lg">
				<ScanLine
					className="mr-2 inline align-middle text-blue-400"
					size={20}
				/>
				<span className="align-middle">Scan Control</span>
			</h2>

			<div className="flex flex-grow flex-col gap-2.5">
				{/* Model Selection */}
				<div>
					<label className="mb-1 flex items-center text-xs font-medium text-gray-300">
						<SquareTerminal
							size={12}
							className="mr-1.5 opacity-80"
						/>{" "}
						YOLO Model
					</label>
					<Select<{ value: string; label: string }, false> // Single selection
						value={selectedModel}
						options={modelOptions}
						onChange={(option) =>
							setSelectedModel(option || modelOptions[0])
						} // Ensure non-null
						isDisabled={scanRunning || isStoreLoading}
						formatOptionLabel={formatOptionLabel}
						styles={selectStyles}
						menuPortalTarget={document.body} // Render menu in portal
						menuPosition="fixed" // Prevent menu clipping
						classNamePrefix="react-select" // Add prefix for potential global styling
					/>
				</div>

				{/* Country Selection */}
				<div>
					<label className="mb-1 flex items-center text-xs font-medium text-gray-300">
						<Globe size={12} className="mr-1.5 opacity-80" />{" "}
						Country{" "}
						<span className="ml-1 text-gray-400">(Optional)</span>
					</label>
					<Select<{ value: string; name: string }, false> // Single selection
						value={selectedCountry}
						options={countryOptions}
						onChange={(option) => setSelectedCountry(option)} // Can be null
						isDisabled={scanRunning || isStoreLoading}
						placeholder="All Countries"
						isClearable
						filterOption={
							(
								option,
								inputValue, // Custom filter for name
							) =>
								option.data.name
									.toLowerCase()
									.includes(inputValue.toLowerCase()) ||
								option.data.value
									.toLowerCase()
									.includes(inputValue.toLowerCase()) // Allow searching by code too
						}
						formatOptionLabel={formatOptionLabel}
						styles={selectStyles}
						menuPortalTarget={document.body}
						menuPosition="fixed"
						classNamePrefix="react-select"
					/>
				</div>
			</div>

			{/* Action Buttons */}
			<div className="mt-auto flex-shrink-0">
				{scanRunning ? (
					<button
						onClick={handleStopScan}
						disabled={isStoreLoading}
						className="flex w-full items-center justify-center rounded-md bg-red-600/90 px-3 py-1.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-red-700 focus-visible:ring-red-500 cursor-pointer disabled:cursor-not-allowed"
					>
						{isStoreLoading ? (
							<Loader2
								className="mr-1.5 animate-spin"
								size={16}
							/>
						) : (
							<StopCircle className="mr-1.5" size={16} />
						)}
						<span className="align-middle">Stop Scan</span>
					</button>
				) : (
					<button
						onClick={handleStartScan}
						disabled={isStoreLoading}
						className="flex w-full cursor-pointer items-center justify-center rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-blue-700 focus-visible:ring-blue-500 disabled:cursor-not-allowed"
					>
						{isStoreLoading && ( // Only show spinner when starting
							<Loader2
								className="mr-1.5 animate-spin"
								size={16}
							/>
						)}
						<span className="align-middle">Start Scan</span>
					</button>
				)}
			</div>
		</Panel>
	);
};

export default ScanPanel;
