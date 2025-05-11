import { useEffect, useState } from "react"
import InteractiveGraph from "./InteractiveGraph"

const tmp = [
    { x: 0.01, y: 0.01 },
    { x: 0.6, y: 0.4 },
    { x: 0.7, y: 0.7 },
	{ x: 0.99, y: 0.99 }
]

const InteractivePage = () => {
	const [points, setPoints] = useState<Point[]>(tmp)

	const [dataPoints, setDataPoints] = useState<Point[]>([]);

	useEffect(() => {
		
	}, [dataPoints])

	return (
		<>
			<InteractiveGraph
				points_input = {points}
				setPoints_input = {setPoints}
				regPoints_output= {dataPoints}
				setRegPoints_output={setDataPoints}
			/>
		</>
	)
}


export default InteractivePage