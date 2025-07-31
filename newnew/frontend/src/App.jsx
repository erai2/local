import React, { useEffect, useState } from "react";
import RuleTable from "./components/RuleTable";
import RuleCards from "./components/RuleCards";
import MindmapViz from "./components/MindmapViz";
import NetworkViz from "./components/NetworkViz";
import ExportMenu from "./components/ExportMenu";

export default function App() {
  const [rules, setRules] = useState([]);
  const [view, setView] = useState("table");

  useEffect(() => {
    fetch("/rules").then(res => res.json()).then(setRules);
  }, []);

  // 파일 업로드로 규칙 자동추출
  const handleExtract = async (e) => {
    const file = e.target.files[0];
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/extract_rules", {method: "POST", body: form});
    const data = await res.json();
    setRules(rules.concat(data.rules));
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto bg-white shadow-lg rounded-lg p-8">
        <h1 className="text-3xl font-bold mb-6 text-blue-700">규칙 관리 시스템</h1>
        <ExportMenu />

        {/* 뷰 전환 버튼 */}
        <div className="flex gap-2 mb-4">
          <button className={`px-4 py-2 rounded ${view === "table" ? "bg-blue-600 text-white" : "bg-gray-200"}`} onClick={() => setView("table")}>표</button>
          <button className={`px-4 py-2 rounded ${view === "card" ? "bg-blue-600 text-white" : "bg-gray-200"}`} onClick={() => setView("card")}>카드</button>
          <button className={`px-4 py-2 rounded ${view === "mindmap" ? "bg-blue-600 text-white" : "bg-gray-200"}`} onClick={() => setView("mindmap")}>마인드맵</button>
          <button className={`px-4 py-2 rounded ${view === "network" ? "bg-blue-600 text-white" : "bg-gray-200"}`} onClick={() => setView("network")}>네트워크</button>
        </div>

        <div className="mb-6">
          {view === "table" && <RuleTable rules={rules} setRules={setRules} />}
          {view === "card" && <RuleCards rules={rules} />}
          {view === "mindmap" && <MindmapViz rules={rules} />}
          {view === "network" && <NetworkViz rules={rules} />}
        </div>

        {/* AI 규칙 자동추출 */}
        <div className="border-t pt-6 mt-6">
          <h3 className="text-lg font-semibold mb-2 text-blue-800">AI 규칙 자동추출 (파일 업로드)</h3>
          <input type="file" className="border rounded px-3 py-1" onChange={handleExtract} />
        </div>
      </div>
    </div>
  );
}
