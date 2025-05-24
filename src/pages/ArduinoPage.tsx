import { useEffect, useState, useRef } from "react";
import { Switch } from "@/components/ui/switch";
import Chart from 'chart.js/auto';
import ClassificationResult from "@/components/ClassificationResult";
import { Button } from "@/components/ui/button";

interface ClassificationResultType {
  results: {
    model: string;
    probabilities: number[];
    prediction: string;
  }[];
}

const ArduinoPage = () => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [heartbeatData, setHeartbeatData] = useState<number[]>([]);
  const [classificationResult, setClassificationResult] = useState<ClassificationResultType | null>(null);
  const [isLoadingResult, setIsLoadingResult] = useState<boolean>(false);
  const chartRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);
  const intervalRef = useRef<number | null>(null);

  // Function to fetch the current heartbeat data
  const fetchHeartbeatData = async () => {
    try {
      const response = await fetch("http://127.0.0.1:5001/api/current_beat");
      console.log("response:", response);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
        console.log("amplitude:",data.amplitude);
      if (data && data.amplitude) {
        setHeartbeatData(data.amplitude);
        // Classify the signal automatically when we get new data
        classifySignal(data.amplitude);
      }
    } catch (error) {
      console.error("Error fetching heartbeat data:", error);
      // Don't stop monitoring on error - we'll try again in 2 seconds
    }
  };

  // Function to classify the heartbeat signal
  const classifySignal = async (signalToClassify: number[]) => {
    setIsLoadingResult(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_BACKED_URL}/classify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ signal: signalToClassify }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      if (data.error) {
        console.error('Classification error:', data.error);
        alert('Lỗi từ server: ' + data.error);
      } else {
        setClassificationResult(data);
      }
    } catch (error) {
      console.error('Classification error:', error);
      alert('Error classifying signal: ' + (error instanceof Error ? error.message : String(error)));
    } finally {
      setIsLoadingResult(false);
    }
  };

  // Handle start/stop monitoring
  useEffect(() => {
    if (isMonitoring) {
      // Fetch data immediately when monitoring starts
      fetchHeartbeatData();
      
      // Set up interval for periodic fetching
      intervalRef.current = window.setInterval(fetchHeartbeatData, 2000);
    } else if (intervalRef.current) {
      // Clear the interval when monitoring stops
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Cleanup function to clear interval when component unmounts
    return () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
      }
    };
  }, [isMonitoring]);
  // Update chart when heartbeat data changes
  useEffect(() => {
    if (!heartbeatData.length || !chartRef.current) return;

    const ctx = chartRef.current.getContext('2d');
    if (!ctx) return;

    // Destroy existing chart if it exists
    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
    }

    // Create new chart
    chartInstanceRef.current = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: heartbeatData.length }, (_, i) => i),
        datasets: [
          {
            label: 'Real-time Heartbeat Signal',
            data: heartbeatData,
            borderColor: '#FF6384',
            borderWidth: 1.5,
            fill: false,
            tension: 0.2, // Adds a slight curve to the line
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 500 // Fast animation for real-time updates
        },
        layout: {
          padding: {
            left: 0,
            right: 0
          }
        },
        scales: {
          x: { 
            title: { display: true, text: 'Time (sample)' },
            grid: {
              color: 'rgba(200, 200, 200, 0.1)'
            }
          },
          y: { 
            title: { display: true, text: 'Amplitude' },
            grid: {
              color: 'rgba(200, 200, 200, 0.1)'
            }
          },
        },
      },
    });

    // Clean up function
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [heartbeatData]);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Arduino ECG Monitoring</h1>
      
      <div className="flex items-center space-x-4">
        <span className="font-medium">Real-time Monitoring:</span>
        <Switch 
          checked={isMonitoring} 
          onCheckedChange={setIsMonitoring} 
          id="monitoring-switch"
        />
        <span className={isMonitoring ? "text-green-500" : "text-gray-500"}>
          {isMonitoring ? "Active" : "Inactive"}
        </span>
      </div>      <div className="bg-card rounded-lg p-4 border border-border">
        <h2 className="text-lg font-semibold mb-2">Real-time Heartbeat Signal</h2>
        <div className="h-[300px] w-full relative">
          <div className="absolute inset-0">
            <canvas ref={chartRef}></canvas>
          </div>
        </div>
      </div>

      {heartbeatData.length > 0 && (
        <Button
          onClick={() => classifySignal(heartbeatData)}
          disabled={isLoadingResult}
          className="mt-4"
        >
          {isLoadingResult ? "Classifying..." : "Classify Current Signal"}
        </Button>
      )}

      {isLoadingResult && (
        <div className="text-yellow-500 animate-pulse">
          Classifying heartbeat signal...
        </div>
      )}

      <div className="bg-card rounded-lg p-4 border border-border">
        <h2 className="text-lg font-semibold mb-2">Classification Result</h2>
        {classificationResult ? (
          <ClassificationResult classificationResult={classificationResult} />
        ) : (
          <p className="text-gray-500 italic">No classification result available yet.</p>
        )}
      </div>
    </div>
  );
};

export default ArduinoPage;