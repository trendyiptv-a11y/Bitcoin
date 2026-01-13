import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  phaseCard: {
    borderRadius: 16,
    padding: 24,
    marginBottom: 20,
    elevation: 8,
    shadowColor: '#00d4ff',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  dateText: {
    color: '#999',
    fontSize: 12,
    marginBottom: 8,
    opacity: 0.8,
  },
  phaseLabel: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 16,
  },
  scoreContainer: {
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.2)',
    paddingTop: 12,
  },
  scoreLabel: {
    color: '#ccc',
    fontSize: 12,
    marginBottom: 4,
  },
  scoreValue: {
    color: '#fff',
    fontSize: 32,
    fontWeight: '700',
  },
  metricsGrid: {
    gap: 12,
    marginBottom: 24,
  },
  metricCard: {
    borderRadius: 12,
    padding: 16,
    elevation: 4,
  },
  metricLabel: {
    color: '#99ccff',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 8,
  },
  metricValue: {
    color: '#fff',
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 6,
  },
  metricDescription: {
    color: '#aaa',
    fontSize: 11,
  },
  section: {
    marginBottom: 20,
  },
  sectionTitle: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 12,
  },
  tensorCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    elevation: 4,
  },
  tensorLabel: {
    color: '#90ee90',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 6,
  },
  tensorValue: {
    color: '#fff',
    fontSize: 28,
    fontWeight: '700',
  },
  componentsGrid: {
    gap: 12,
  },
  componentCard: {
    borderRadius: 12,
    padding: 14,
    elevation: 4,
  },
  componentLabel: {
    color: '#ffaa66',
    fontSize: 12,
    fontWeight: '500',
    marginBottom: 6,
  },
  componentValue: {
    color: '#fff',
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 10,
  },
  subcomponents: {
    gap: 4,
  },
  subLabel: {
    color: '#bbb',
    fontSize: 11,
  },
  footer: {
    height: 40,
  },
});
