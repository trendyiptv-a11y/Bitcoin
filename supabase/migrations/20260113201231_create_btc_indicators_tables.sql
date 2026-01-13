/*
  # Create BTC Indicators Tables

  1. New Tables
    - `ic_btc_snapshots` - Stores IC_BTC structural and flux indicators
    - `j_btc_snapshots` - Stores J_BTC network tension metrics
  
  2. Security
    - Enable RLS on both tables
    - Add policies for public read access (indicators are public data)
  
  3. Indexes
    - Add date indexes for efficient querying of recent data
*/

CREATE TABLE IF NOT EXISTS ic_btc_snapshots (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  date date NOT NULL UNIQUE,
  icc numeric NOT NULL,
  ic_struct numeric NOT NULL,
  ic_flux numeric NOT NULL,
  mega_score numeric NOT NULL,
  mega_phase_code text NOT NULL,
  mega_phase_label text NOT NULL,
  structure_label text NOT NULL,
  direction_label text NOT NULL,
  subcycles_label text NOT NULL,
  show_base_card boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS j_btc_snapshots (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  timestamp_utc timestamptz NOT NULL UNIQUE,
  j_tot numeric NOT NULL,
  j_tech numeric NOT NULL,
  j_soc numeric NOT NULL,
  tech_c_hash numeric NOT NULL,
  tech_c_nodes numeric NOT NULL,
  tech_c_mempool numeric NOT NULL,
  soc_c_custody numeric NOT NULL,
  soc_c_gov numeric NOT NULL,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

ALTER TABLE ic_btc_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE j_btc_snapshots ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public read access to IC_BTC snapshots"
  ON ic_btc_snapshots FOR SELECT
  TO public
  USING (true);

CREATE POLICY "Public read access to J_BTC snapshots"
  ON j_btc_snapshots FOR SELECT
  TO public
  USING (true);

CREATE INDEX idx_ic_btc_date ON ic_btc_snapshots(date DESC);
CREATE INDEX idx_j_btc_timestamp ON j_btc_snapshots(timestamp_utc DESC);
