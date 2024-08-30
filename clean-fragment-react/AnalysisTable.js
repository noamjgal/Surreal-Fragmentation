import React from 'react';

const analysisData = [
  {
    analysis: 'T-test (Stationary)',
    fragmentation: 'Stationary_index',
    emotion: 'WORRY',
    statistic: 2.317,
    pValue: 0.022,
    direction: 'High: 2.30, Low: 1.93'
  },
  {
    analysis: 'T-test (Mobile)',
    fragmentation: 'Mobile_index',
    emotion: 'WORRY',
    statistic: 2.053,
    pValue: 0.043,
    direction: 'High: 2.27, Low: 1.92'
  }
];

const tableStyle = {
  width: '100%',
  borderCollapse: 'collapse',
  fontFamily: 'Arial, sans-serif',
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

const AnalysisTable = () => {
  const features = ['Analysis', 'Fragmentation', 'Emotion', 'Statistic', 'P-value', 'Direction'];

  return (
    <div style={{ padding: '20px' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>Analysis Results</h2>
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={headerCellStyle}>Feature</th>
            {analysisData.map((data, index) => (
              <th key={index} style={headerCellStyle}>{`Test ${index + 1}`}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((feature, index) => (
            <tr key={feature} style={{ backgroundColor: index % 2 === 0 ? '#ffffff' : '#f9fafb' }}>
              <td style={headerCellStyle}>{feature}</td>
              {analysisData.map((data, dataIndex) => (
                <td key={dataIndex} style={cellStyle}>
                  {feature === 'Analysis' && data.analysis}
                  {feature === 'Fragmentation' && data.fragmentation}
                  {feature === 'Emotion' && data.emotion}
                  {feature === 'Statistic' && data.statistic.toFixed(3)}
                  {feature === 'P-value' && data.pValue.toFixed(3)}
                  {feature === 'Direction' && data.direction}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AnalysisTable;