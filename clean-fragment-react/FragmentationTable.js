import React from 'react';

const data = [
  {
    metric: 'Stationary_index',
    medianCutoff: 0.85,
    population: { high: 55, low: 58 },
    mean: { high: 0.91, low: 0.65 },
    median: { high: 0.92, low: 0.71 },
    stdDev: { high: 0.04, low: 0.19 },
    min: { high: 0.86, low: 0.0 },
    max: { high: 0.97, low: 0.85 },
    totalMobileDuration: { high: 4641.05, low: 3602.12 }
  },
  {
    metric: 'Mobile_index',
    medianCutoff: 0.94,
    population: { high: 47, low: 49 },
    mean: { high: 0.99, low: 0.39 },
    median: { high: 0.99, low: 0.0 },
    stdDev: { high: 0.02, low: 0.42 },
    min: { high: 0.95, low: 0.0 },
    max: { high: 1.0, low: 0.94 },
    totalMobileDuration: { high: 3877.42, low: 4365.75 }
  },
  // ... (other data rows)
];

const tableStyle = {
  width: '100%',
  borderCollapse: 'collapse',
  fontFamily: 'Arial, sans-serif',
  marginBottom: '30px',
};

const cellStyle = {
  padding: '12px 15px',
  textAlign: 'center',
  border: '1px solid #dee2e6',
};

const headerCellStyle = {
  ...cellStyle,
  backgroundColor: '#f3f4f6',
  fontWeight: 'bold',
};

const FragmentationTable = () => {
  const features = [
    { header: 'Median Cut-off', getValue: row => row.medianCutoff },
    { header: 'Population', getValue: row => row.population },
    { header: 'Mean', getValue: row => row.mean },
    { header: 'Median', getValue: row => row.median },
    { header: 'Std Dev', getValue: row => row.stdDev },
    { header: 'Min', getValue: row => row.min },
    { header: 'Max', getValue: row => row.max },
    { header: 'Total Mobile Duration', getValue: row => row.totalMobileDuration },
  ];

  const formatValue = (value, feature) => {
    if (feature.header === 'Median Cut-off') {
      return value.toFixed(2);
    }
    return `High: ${value.high.toFixed(2)}, Low: ${value.low.toFixed(2)}`;
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>Fragmentation Data</h2>
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={headerCellStyle}>Feature</th>
            {data.map(row => (
              <th key={row.metric} style={headerCellStyle}>{row.metric}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((feature, index) => (
            <tr key={feature.header} style={{ backgroundColor: index % 2 === 0 ? '#ffffff' : '#f9fafb' }}>
              <td style={headerCellStyle}>{feature.header}</td>
              {data.map(row => (
                <td key={row.metric} style={cellStyle}>
                  {formatValue(feature.getValue(row), feature)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default FragmentationTable;