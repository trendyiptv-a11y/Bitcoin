const API_BASE = 'http://localhost:8000';

export interface ICBTCMega {
  date: string;
  icc: number;
  ic_struct: number;
  ic_flux: number;
  mega_score: number;
  mega_phase_code: string;
  mega_phase_label: string;
  structure_label: string;
  direction_label: string;
  subcycles_label: string;
  show_base_card: boolean;
}

export interface JBTCLatest {
  timestamp_utc: string;
  J_tot: number;
  J_tech: number;
  J_soc: number;
  tech_C_hash: number;
  tech_C_nodes: number;
  tech_C_mempool: number;
  soc_C_custody: number;
  soc_C_gov: number;
}

export async function fetchICBTCMega(): Promise<ICBTCMega> {
  const response = await fetch(`${API_BASE}/v1/ic_btc/mega`);
  if (!response.ok) throw new Error('Failed to fetch IC_BTC data');
  return response.json();
}

export async function fetchJBTCLatest(): Promise<JBTCLatest> {
  const response = await fetch(`${API_BASE}/v1/ic_btc/latest`);
  if (!response.ok) throw new Error('Failed to fetch J_BTC data');
  return response.json();
}

export async function fetchICBTCHistory() {
  const response = await fetch(`${API_BASE}/v1/ic_btc/history`);
  if (!response.ok) throw new Error('Failed to fetch history');
  return response.json();
}
