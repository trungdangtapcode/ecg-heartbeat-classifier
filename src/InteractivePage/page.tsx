import { useEffect, useRef, useState } from "react"
import InteractiveGraph, { type InteractiveGraphRef } from "./InteractiveGraph"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Button } from "@/components/ui/button"

const tmp = [
    { x: 0.01, y: 0.01 },
    { x: 0.6, y: 0.4 },
    { x: 0.7, y: 0.7 },
    { x: 0.99, y: 0.99 }
]

interface Point {
    x: number;
    y: number;
}

interface AnchorData {
    name: string;
    anchor: Point[];
    label: number;
}

const InteractivePage = () => {
    const [points, setPoints] = useState<Point[]>(tmp)
    const [dataPoints, setDataPoints] = useState<Point[]>([])
    const [anchorData, setAnchorData] = useState<AnchorData[]>([])
    const [selectedAnchor, setSelectedAnchor] = useState<string>("")

    const graphRef = useRef<InteractiveGraphRef>(null)

    // Fetch anchor data on component mount
    useEffect(() => {
        fetch("/sample_anchor.json")
            .then(response => response.json())
            .then(data => setAnchorData(data))
            .catch(error => console.error("Error fetching anchor data:", error))
    }, [])

    // Update graph when dataPoints change
    // useEffect(() => {
    //     console.log("points", JSON.stringify(points))
    //     graphRef.current?.setPointsFromInput()
    // }, [dataPoints])

    // Handle loading selected anchor
    const handleLoadAnchor = () => {
        if (selectedAnchor) {
            const selectedData = anchorData.find(data => data.name === selectedAnchor)
            if (selectedData?.anchor) {
                setPoints(selectedData.anchor)
                graphRef.current?.setPointsFromInput(selectedData.anchor)
            }
        }
    }

    return (
        <div className="p-4 space-y-4">
            <div className="flex items-center gap-2">
                <Select 
                    value={selectedAnchor} 
                    onValueChange={setSelectedAnchor}
                >
                    <SelectTrigger className="w-[200px]">
                        <SelectValue placeholder="Select anchor data" />
                    </SelectTrigger>
                    <SelectContent>
                        {anchorData.map((data) => (
                            <SelectItem key={data.name} value={data.name}
							className="cursor-pointer">
                                {data.name || "Unnamed Anchor"}
                            </SelectItem>
                        ))}
                    </SelectContent>
                </Select>
                <Button 
                    onClick={handleLoadAnchor}
                    disabled={!selectedAnchor}
                >
                    Load Selected Anchor
                </Button>
            </div>
            <InteractiveGraph
                ref={graphRef}
                points_input={points}
                setPoints_input={setPoints}
                regPoints_output={dataPoints}
                setRegPoints_output={setDataPoints}
            />
        </div>
    )
}

export default InteractivePage