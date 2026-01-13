import React, { useEffect, useState } from 'react';
import { View, Text, ScrollView, ActivityIndicator, RefreshControl } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { fetchICBTCMega, fetchJBTCLatest, ICBTCMega, JBTCLatest } from '@/lib/api';
import { styles } from '@/styles/dashboard.styles';

export default function Dashboard() {
  const [icBtc, setIcBtc] = useState<ICBTCMega | null>(null);
  const [jBtc, setJBtc] = useState<JBTCLatest | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadData = async () => {
    try {
      const [ic, j] = await Promise.all([
        fetchICBTCMega(),
        fetchJBTCLatest(),
      ]);
      setIcBtc(ic);
      setJBtc(j);
    } catch (error) {
      console.error('Error loading data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const onRefresh = async () => {
    setRefreshing(true);
    await loadData();
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#00d4ff" />
      </View>
    );
  }

  const getPhaseColor = (phaseCode: string) => {
    const colors: Record<string, string[]> = {
      late_bull: ['#1a4d2e', '#2d7a3d'],
      early_bear: ['#8b0000', '#cc0000'],
      structural_dips: ['#4a5899', '#5a7cbc'],
      ranging: ['#664d00', '#996600'],
    };
    return colors[phaseCode] || ['#1a1a2e', '#16213e'];
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {icBtc && (
        <LinearGradient
          colors={getPhaseColor(icBtc.mega_phase_code)}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.phaseCard}
        >
          <Text style={styles.dateText}>{icBtc.date}</Text>
          <Text style={styles.phaseLabel}>{icBtc.mega_phase_label}</Text>
          <View style={styles.scoreContainer}>
            <Text style={styles.scoreLabel}>Scor Mega</Text>
            <Text style={styles.scoreValue}>{icBtc.mega_score.toFixed(1)}</Text>
          </View>
        </LinearGradient>
      )}

      <View style={styles.metricsGrid}>
        {icBtc && (
          <>
            <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.metricCard}>
              <Text style={styles.metricLabel}>IC Structură</Text>
              <Text style={styles.metricValue}>{icBtc.ic_struct.toFixed(2)}</Text>
              <Text style={styles.metricDescription}>{icBtc.structure_label}</Text>
            </LinearGradient>

            <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.metricCard}>
              <Text style={styles.metricLabel}>IC Flux</Text>
              <Text style={styles.metricValue}>{icBtc.ic_flux.toFixed(2)}</Text>
              <Text style={styles.metricDescription}>{icBtc.direction_label}</Text>
            </LinearGradient>

            <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.metricCard}>
              <Text style={styles.metricLabel}>ICC</Text>
              <Text style={styles.metricValue}>{icBtc.icc.toFixed(1)}</Text>
              <Text style={styles.metricDescription}>{icBtc.subcycles_label}</Text>
            </LinearGradient>
          </>
        )}
      </View>

      {jBtc && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Metrici Rețea (J_BTC)</Text>

          <LinearGradient colors={['#1a2d1a', '#2d4d2d']} style={styles.tensorCard}>
            <Text style={styles.tensorLabel}>J Total</Text>
            <Text style={styles.tensorValue}>{jBtc.J_tot.toFixed(4)}</Text>
          </LinearGradient>

          <View style={styles.componentsGrid}>
            <LinearGradient colors={['#1a1a2e', '#16213e']} style={styles.componentCard}>
              <Text style={styles.componentLabel}>Tehnologic</Text>
              <Text style={styles.componentValue}>{jBtc.J_tech.toFixed(4)}</Text>
              <View style={styles.subcomponents}>
                <Text style={styles.subLabel}>Hash: {jBtc.tech_C_hash.toFixed(2)}</Text>
                <Text style={styles.subLabel}>Noduri: {jBtc.tech_C_nodes.toFixed(2)}</Text>
                <Text style={styles.subLabel}>Mempool: {jBtc.tech_C_mempool.toFixed(3)}</Text>
              </View>
            </LinearGradient>

            <LinearGradient colors={['#2d1a1a', '#4d2d2d']} style={styles.componentCard}>
              <Text style={styles.componentLabel}>Social</Text>
              <Text style={styles.componentValue}>{jBtc.J_soc.toFixed(4)}</Text>
              <View style={styles.subcomponents}>
                <Text style={styles.subLabel}>Custody: {jBtc.soc_C_custody.toFixed(2)}</Text>
                <Text style={styles.subLabel}>Governance: {jBtc.soc_C_gov.toFixed(2)}</Text>
              </View>
            </LinearGradient>
          </View>
        </View>
      )}

      <View style={styles.footer} />
    </ScrollView>
  );
}
