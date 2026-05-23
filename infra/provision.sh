#!/usr/bin/env bash
# Provisions Azure resources for SIT788 11.2HD Recipe Recommender.
#
# Lab subscription notes (Deakin labs DS - 1208):
#   - The user has Cognitive Services Contributor role at sub scope; cannot
#     create new resource groups. We reuse the pre-existing "deakinuni" RG.
#   - An existing Computer Vision (F0) account 'sit788cv2srbwq' is reused.
#   - Speech + OpenAI accounts are created fresh in this RG.
#   - Model versions: gpt-5-mini (2025-08-07) + text-embedding-3-small.
#     gpt-4o-mini was deprecated 2026-03-31 and cannot be deployed.

set -euo pipefail

RG=deakinuni                              # existing lab RG
STUDENT_ID=s225048421
VISION_NAME=sit788cv2srbwq                # pre-existing CV F0 account, reused
SPEECH_NAME=sit788-${STUDENT_ID}-speech
OPENAI_NAME=sit788-${STUDENT_ID}-openai
LOC_AU=australiaeast
LOC_US=eastus

echo "Verifying existing Vision resource '$VISION_NAME'..."
az cognitiveservices account show -g "$RG" -n "$VISION_NAME" --query "kind" -o tsv

echo "Creating Azure Speech..."
az cognitiveservices account create \
  -g "$RG" -n "$SPEECH_NAME" \
  --kind SpeechServices --sku S0 -l "$LOC_AU" --yes >/dev/null

echo "Creating Azure OpenAI..."
az cognitiveservices account create \
  -g "$RG" -n "$OPENAI_NAME" \
  --kind OpenAI --sku S0 -l "$LOC_US" --yes >/dev/null

echo "Deploying gpt-5-mini..."
az cognitiveservices account deployment create \
  -g "$RG" -n "$OPENAI_NAME" \
  --deployment-name gpt-5-mini \
  --model-name gpt-5-mini --model-version "2025-08-07" \
  --model-format OpenAI --sku-capacity 30 --sku-name "GlobalStandard"

echo "Deploying text-embedding-3-small..."
az cognitiveservices account deployment create \
  -g "$RG" -n "$OPENAI_NAME" \
  --deployment-name text-embedding-3-small \
  --model-name text-embedding-3-small --model-version "1" \
  --model-format OpenAI --sku-capacity 30 --sku-name "Standard"

echo
echo "==== Copy these into your .env ===="
echo "AZURE_VISION_ENDPOINT=$(az cognitiveservices account show -g "$RG" -n "$VISION_NAME" --query properties.endpoint -o tsv)"
echo "AZURE_VISION_KEY=$(az cognitiveservices account keys list -g "$RG" -n "$VISION_NAME" --query key1 -o tsv)"
echo "AZURE_SPEECH_KEY=$(az cognitiveservices account keys list -g "$RG" -n "$SPEECH_NAME" --query key1 -o tsv)"
echo "AZURE_SPEECH_REGION=$LOC_AU"
echo "AZURE_OPENAI_ENDPOINT=$(az cognitiveservices account show -g "$RG" -n "$OPENAI_NAME" --query properties.endpoint -o tsv)"
echo "AZURE_OPENAI_KEY=$(az cognitiveservices account keys list -g "$RG" -n "$OPENAI_NAME" --query key1 -o tsv)"
echo "AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-5-mini"
echo "AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small"
