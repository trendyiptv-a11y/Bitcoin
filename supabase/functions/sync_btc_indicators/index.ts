import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2.38.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Client-Info, Apikey",
};

Deno.serve(async (req: Request) => {
  if (req.method === "OPTIONS") {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const supabaseKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY");
    const apiUrl = Deno.env.get("BTC_API_URL") || "http://localhost:8000";

    const supabase = createClient(supabaseUrl!, supabaseKey!);

    // Fetch IC_BTC data
    const icResponse = await fetch(`${apiUrl}/v1/ic_btc/mega`);
    const icData = await icResponse.json();

    if (icData && icData.date) {
      await supabase.from("ic_btc_snapshots").upsert(
        {
          date: icData.date,
          icc: icData.icc,
          ic_struct: icData.ic_struct,
          ic_flux: icData.ic_flux,
          mega_score: icData.mega_score,
          mega_phase_code: icData.mega_phase_code,
          mega_phase_label: icData.mega_phase_label,
          structure_label: icData.structure_label,
          direction_label: icData.direction_label,
          subcycles_label: icData.subcycles_label,
          show_base_card: icData.show_base_card,
          updated_at: new Date().toISOString(),
        },
        { onConflict: "date" }
      );
    }

    // Fetch J_BTC data
    const jResponse = await fetch(`${apiUrl}/v1/ic_btc/latest`);
    const jData = await jResponse.json();

    if (jData && jData.timestamp_utc) {
      await supabase.from("j_btc_snapshots").upsert(
        {
          timestamp_utc: jData.timestamp_utc,
          j_tot: jData.J_tot,
          j_tech: jData.J_tech,
          j_soc: jData.J_soc,
          tech_c_hash: jData.tech_C_hash,
          tech_c_nodes: jData.tech_C_nodes,
          tech_c_mempool: jData.tech_C_mempool,
          soc_c_custody: jData.soc_C_custody,
          soc_c_gov: jData.soc_C_gov,
          updated_at: new Date().toISOString(),
        },
        { onConflict: "timestamp_utc" }
      );
    }

    return new Response(
      JSON.stringify({
        success: true,
        message: "Data synced successfully",
        ic_updated: !!icData?.date,
        j_updated: !!jData?.timestamp_utc,
      }),
      {
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      }
    );
  } catch (error) {
    return new Response(
      JSON.stringify({
        success: false,
        error: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: {
          ...corsHeaders,
          "Content-Type": "application/json",
        },
      }
    );
  }
});
