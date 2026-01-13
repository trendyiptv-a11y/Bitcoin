import React, { useEffect, useState } from 'react';
import { View, Text, ScrollView, ActivityIndicator } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { fetchICBTCHistory } from '@/lib/api';
import { styles } from '@/styles/indicators.styles';

interface HistoryEntry {
  date: string;
  ic_struct: number;
  ic_flux: number;
  icc: number;
  mega_phase_code: string;
  mega_score: number;
}

export default function Indicators() {
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const data = await fetchICBTCHistory();
        if (Array.isArray(data)) {
          setHistory(data.slice(-30));
        }
      } catch (error) {
        console.error('Error loading history:', error);
      } finally {
        setLoading(false);
      }
    };

    loadHistory();
  }, []);

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#00d4ff" />
      </View>
    );
  }

  const getPhaseColor = (phaseCode: string) => {
    const colors: Record<string, string> = {
      late_bull: '#4CAF50',
      early_bear: '#FF5252',
      structural_dips: '#2196F3',
      ranging: '#FF9800',
    };
    return colors[phaseCode] || '#999';
  };

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Istorie Indicatori (Ultimele 30 zile)</Text>

      <View style={styles.tableHeader}>
        <Text style={[styles.tableCell, styles.headerText]}>Data</Text>
        <Text style={[styles.tableCell, styles.headerText]}>Scor</Text>
        <Text style={[styles.tableCell, styles.headerText]}>Structură</Text>
        <Text style={[styles.tableCell, styles.headerText]}>Flux</Text>
      </View>

      {history.map((entry, idx) => (
        <LinearGradient
          key={idx}
          colors={['#1a1a2e', '#0f0f1e']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
          style={styles.row}
        >
          <Text style={[styles.tableCell, styles.dateCell]}>{entry.date}</Text>
          <View
            style={[
              styles.scoreCell,
              { borderLeftColor: getPhaseColor(entry.mega_phase_code) },
            ]}
          >
            <Text style={styles.scoreText}>{entry.mega_score.toFixed(1)}</Text>
          </View>
          <Text style={[styles.tableCell, styles.valueCell]}>
            {entry.ic_struct.toFixed(1)}
          </Text>
          <Text style={[styles.tableCell, styles.valueCell]}>
            {entry.ic_flux.toFixed(1)}
          </Text>
        </LinearGradient>
      ))}

      <View style={styles.footer} />
    </ScrollView>
  );
}
