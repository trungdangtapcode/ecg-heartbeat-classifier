import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface ModelResult {
  model: string;
  accuracy: string;
  macroF1: string;
  macroPrecision?: string;
  macroRecall?: string;
  weightedF1?: string;
}

interface ResultsTableProps {
  results: ModelResult[];
  title: string;
  description?: string;
}

const ResultsTable = ({ results, title, description }: ResultsTableProps) => {
  // Find maximum values for highlighting
  const maxAccuracy = Math.max(...results.map((row) => parseFloat(row.accuracy)));
  const maxMacroF1 = Math.max(...results.map((row) => parseFloat(row.macroF1)));
  const maxMacroPrecision = results[0]?.macroPrecision 
    ? Math.max(...results.map((row) => parseFloat(row.macroPrecision || "0")))
    : 0;
  const maxMacroRecall = results[0]?.macroRecall
    ? Math.max(...results.map((row) => parseFloat(row.macroRecall || "0")))
    : 0;
  return (
    <div className="mt-6 mb-8">
      <h3 className="text-xl font-medium mb-3 text-[#FFD700]">{title}</h3>
      {description && <p className="mb-4 text-sm sm:text-base text-gray-300">{description}</p>}
      
      {/* Desktop and tablet view */}
      <div className="hidden sm:block overflow-x-auto rounded-lg border border-[#555]">
        <Table>
          <TableHeader className="bg-[#444]">
            <TableRow>
              <TableHead className="text-[#FFD700]">Model</TableHead>
              <TableHead className="text-[#FFD700] text-center">Accuracy</TableHead>
              {results[0]?.macroPrecision && (
                <TableHead className="text-[#FFD700] text-center">Precision (Macro)</TableHead>
              )}
              {results[0]?.macroRecall && (
                <TableHead className="text-[#FFD700] text-center">Recall (Macro)</TableHead>
              )}
              <TableHead className="text-[#FFD700] text-center">F1-Score (Macro)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {results.map((row, index) => (
              <TableRow key={index} className={index % 2 === 0 ? "bg-[#3A3A3A]" : "bg-[#333]"}>
                <TableCell className="font-medium text-gray-200">{row.model}</TableCell>
                <TableCell className={`text-center ${parseFloat(row.accuracy) === maxAccuracy ? "font-bold text-yellow-300" : "text-gray-200"}`}>
                  {row.accuracy}
                </TableCell>
                {row.macroPrecision && (
                  <TableCell className={`text-center ${parseFloat(row.macroPrecision) === maxMacroPrecision ? "font-bold text-yellow-300" : "text-gray-200"}`}>
                    {row.macroPrecision}
                  </TableCell>
                )}
                {row.macroRecall && (
                  <TableCell className={`text-center ${parseFloat(row.macroRecall) === maxMacroRecall ? "font-bold text-yellow-300" : "text-gray-200"}`}>
                    {row.macroRecall}
                  </TableCell>
                )}
                <TableCell className={`text-center ${parseFloat(row.macroF1) === maxMacroF1 ? "font-bold text-yellow-300" : "text-gray-200"}`}>
                  {row.macroF1}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      
      {/* Mobile view - Cards instead of table */}
      <div className="sm:hidden space-y-4">
        {results.map((row, index) => (
          <div key={index} className={`p-4 rounded-lg border border-[#555] ${index % 2 === 0 ? "bg-[#3A3A3A]" : "bg-[#333]"}`}>
            <h4 className="font-medium text-gray-200 border-b border-[#555] pb-2 mb-2">{row.model}</h4>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="text-gray-400">Accuracy:</div>
              <div className={`text-right ${parseFloat(row.accuracy) === maxAccuracy ? "font-bold text-yellow-300" : "text-gray-200"}`}>
                {row.accuracy}
              </div>
              
              {row.macroPrecision && (
                <>
                  <div className="text-gray-400">Precision:</div>
                  <div className={`text-right ${parseFloat(row.macroPrecision) === maxMacroPrecision ? "font-bold text-yellow-300" : "text-gray-200"}`}>
                    {row.macroPrecision}
                  </div>
                </>
              )}
              
              {row.macroRecall && (
                <>
                  <div className="text-gray-400">Recall:</div>
                  <div className={`text-right ${parseFloat(row.macroRecall) === maxMacroRecall ? "font-bold text-yellow-300" : "text-gray-200"}`}>
                    {row.macroRecall}
                  </div>
                </>
              )}
              
              <div className="text-gray-400">F1-Score:</div>
              <div className={`text-right ${parseFloat(row.macroF1) === maxMacroF1 ? "font-bold text-yellow-300" : "text-gray-200"}`}>
                {row.macroF1}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ResultsTable;
