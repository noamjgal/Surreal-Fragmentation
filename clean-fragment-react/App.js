import React from 'react';
import FragmentationTable from './FragmentationTable';
import AnalysisTable from './AnalysisTable';
import EmotionTable from './EmotionTable';

function App() {
  return (
    <div className="App">
      <h1>Fragmentation Analysis</h1>
      <h2>Fragmentation Data</h2>
      <FragmentationTable />
      <h2>Analysis Results</h2>
      <AnalysisTable />
      <h2>Emotion Analysis</h2>
      <EmotionTable />
    </div>
  );
}

export default App;