# CohesivX BTC Monitor — Expo Android Wrapper

This folder contains the Expo wrapper for publishing CohesivX BTC Monitor on Google Play.

## App details

- App name: CohesivX BTC Monitor
- Android package: `app.cohesivx.btcmonitor`
- Version name: `0.1.0`
- Version code: `1`
- Main URL: `https://coezivx.vercel.app/btc-swing-strategy/mecanism.html`
- Privacy Policy: `https://coezivx.vercel.app/btc-swing-strategy/privacy/`
- Permissions: `INTERNET` only

## Local setup

```bash
cd cohesivx-btc-monitor-expo
npm install
```

## Run locally

```bash
npx expo start
```

## Configure EAS

```bash
npm install -g eas-cli
eas login
eas build:configure
```

When EAS asks about Android credentials, choose to generate a new keystore and let EAS manage it.

## Build production AAB for Google Play

```bash
eas build --platform android --profile production
```

The production profile in `eas.json` builds an Android App Bundle (`.aab`) suitable for Google Play.

## Google Play listing

Use this Privacy Policy URL:

```text
https://coezivx.vercel.app/btc-swing-strategy/privacy/
```

Recommended category: Tools.

Important disclaimer:

CohesivX BTC Monitor is an experimental structural monitoring tool. It does not provide financial advice, does not execute transactions and does not manage funds.
