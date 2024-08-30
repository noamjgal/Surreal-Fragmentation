import React from 'react';

const emotionData = [
  { fragmentation: 'Stationary_index', emotion: 'PEACE', highMean: 2.625, lowMean: 2.754, tStatistic: -0.774, pValue: 0.441 },
  { fragmentation: 'Mobile_index', emotion: 'PEACE', highMean: 2.646, lowMean: 2.688, tStatistic: -0.228, pValue: 0.820 },
  { fragmentation: 'Stationary_index', emotion: 'TENSE', highMean: 2.500, lowMean: 2.509, tStatistic: -0.053, pValue: 0.958 },
  { fragmentation: 'Mobile_index', emotion: 'TENSE', highMean: 2.396, lowMean: 2.604, tStatistic: -1.192, pValue: 0.236 },
  // ... (other emotion data)
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

const EmotionTable = () => {
  const features = ['Fragmentation', 'Emotion', 'High Mean', 'Low Mean', 'T Statistic', 'P-value'];
  const uniqueFragmentations = [...new Set(emotionData.map(item => item.fragmentation))];
  const uniqueEmotions = [...new Set(emotionData.map(item => item.emotion))];

  return (
    <div style={{ padding: '20px' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '20px' }}>Emotion Analysis</h2>
      <table style={tableStyle}>
        <thead>
          <tr>
            <th style={headerCellStyle}>Feature</th>
            {uniqueFragmentations.map(frag => (
              uniqueEmotions.map(emotion => (
                <th key={`${frag}-${emotion}`} style={headerCellStyle}>{`${frag} - ${emotion}`}</th>
              ))
            ))}
          </tr>
        </thead>
        <tbody>
          {features.map((feature, index) => (
            <tr key={feature} style={{ backgroundColor: index % 2 === 0 ? '#ffffff' : '#f9fafb' }}>
              <td style={headerCellStyle}>{feature}</td>
              {emotionData.map((data, dataIndex) => (
                <td key={dataIndex} style={cellStyle}>
                  {feature === 'Fragmentation' && data.fragmentation}
                  {feature === 'Emotion' && data.emotion}
                  {feature === 'High Mean' && data.highMean.toFixed(3)}
                  {feature === 'Low Mean' && data.lowMean.toFixed(3)}
                  {feature === 'T Statistic' && data.tStatistic.toFixed(3)}
                  {feature === 'P-value' && data.pValue.toFixed(3)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default EmotionTable;