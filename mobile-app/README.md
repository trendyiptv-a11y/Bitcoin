# CohesivX BTC Mobile App

Aplicație mobilă Android pentru monitorizarea indicatorilor Bitcoin structurali și metrici de rețea.

## Caracteristici

- **Dashboard**: Vizualizare în timp real a indicatorilor IC_BTC și J_BTC
- **Indicatori**: Istorie detaliată a indicatorilor (ultmele 30 zile)
- **Metrici Rețea**: Tensiune tehnologică și socială
- **Mod Întunecat**: Interfață optimizată cu design modern

## Setup

1. Instalează dependențele:
```bash
npm install
```

2. Configurează variabilele de mediu în `.env.local`:
```
EXPO_PUBLIC_SUPABASE_URL=...
EXPO_PUBLIC_SUPABASE_ANON_KEY=...
```

3. Pornește aplicația:
```bash
npm start
```

4. Pentru Android:
```bash
npm run android
```

## Structură Proiect

- `app/(tabs)/` - Ecranele principale (Dashboard, Indicatori, Setări)
- `lib/` - API calls și servicii
- `styles/` - Stylesheet-uri per ecran

## Build Android

```bash
npm run build:android
```

Aceasta creează un APK gata pentru instalare pe dispozitiv Android.
