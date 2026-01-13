import { StyleSheet } from 'react-native';

export const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0a',
    paddingHorizontal: 12,
    paddingVertical: 16,
  },
  title: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  tableHeader: {
    flexDirection: 'row',
    paddingHorizontal: 12,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
    marginBottom: 8,
  },
  tableCell: {
    flex: 1,
    fontSize: 12,
  },
  headerText: {
    color: '#99ccff',
    fontWeight: '600',
  },
  row: {
    flexDirection: 'row',
    paddingHorizontal: 12,
    paddingVertical: 12,
    marginBottom: 8,
    borderRadius: 8,
    alignItems: 'center',
  },
  dateCell: {
    color: '#ccc',
    flex: 1,
  },
  scoreCell: {
    flex: 1,
    borderLeftWidth: 3,
    paddingLeft: 8,
  },
  scoreText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 13,
  },
  valueCell: {
    color: '#aaa',
    flex: 1,
    textAlign: 'right',
    paddingRight: 8,
  },
  footer: {
    height: 40,
  },
});
