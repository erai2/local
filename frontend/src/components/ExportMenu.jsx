export default function ExportMenu() {
  const exportFile = (fmt) => window.open(`/export?fmt=${fmt}`);
  return (
    <div className="mb-4 flex gap-2">
      <button onClick={()=>exportFile("json")} className="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300">JSON 내보내기</button>
      <button onClick={()=>exportFile("excel")} className="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300">Excel 내보내기</button>
      {/* PDF는 구현 시 추가 */}
    </div>
  );
}
