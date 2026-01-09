# processa_ascan_cfrp.py
# Pipeline automático: A-scan -> V_L, V_T -> C_33, C_44 (CFRP)
# Autor: Marcio + Copilot (END)
# Requisitos: Python 3, numpy, pandas, matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def rms_envelope(x, win_samples=401):
	"""Envelope por RMS móvel + janela de Hann (win_samples ímpar)."""
	win = np.hanning(win_samples)
	win = win / win.sum()
	x2 = x**2
	env = np.sqrt(np.convolve(x2, win, mode='same'))
	return env

def find_peak_in_gate(t, env, t_start, t_end):
	"""Retorna (t_pico, idx) do máximo local do envelope no intervalo [t_start, t_end]."""
	mask = (t >= t_start) & (t <= t_end)
	if not np.any(mask):
		return None, None
	idx_seg = np.argmax(env[mask])
	idx_global = np.where(mask)[0][0] + idx_seg
	return t[idx_global], idx_global

def main(args):
	# --- Leitura do CSV ---
	df = pd.read_csv(args.csv)
	# Espera colunas: 't' e 'amp' (ajuste conforme seu arquivo)
	# Se os nomes forem diferentes, substitua abaixo:
	if 't' in df.columns:
		t = df['t'].values
	else:
		t = df.iloc[:,0].values
	if 'amp' in df.columns:
		amp = df['amp'].values
	else:
		amp = df.iloc[:,1].values

	# --- Normalização e envelope ---
	amp = amp - np.mean(amp)
	amp = amp / (np.max(np.abs(amp)) + 1e-12)
	env = rms_envelope(amp, win_samples=args.win)
	# Ajusta o envelope para ter o mesmo tamanho de t
	if len(env) > len(t):
		env = env[:len(t)]
	elif len(env) < len(t):
		env = np.pad(env, (0, len(t)-len(env)), mode='edge')

	# --- Parâmetros físicos ---
	h = float(args.espessura_h) # espessura [m]
	rho = float(args.densidade) # densidade [kg/m3]
	vL0 = float(args.vL0) # estimativa inicial [m/s]
	vT0 = float(args.vT0) # estimativa inicial [m/s]

	# --- Gates (pulso-eco pela espessura) ---
	# Pulso inicial (t0): tomamos o início do registro como referência
	t0 = t[np.argmax(env[:max(10, args.win)])] if len(t) > args.win else t[0]

	# Janela para eco L–L (volta 2h)
	tLL_est = t0 + 2*h / vL0
	gate_LL = (tLL_est - args.margem, tLL_est + args.margem)

	# Janela para eco T–T
	tTT_est = t0 + 2*h / vT0
	gate_TT = (tTT_est - args.margem, tTT_est + args.margem)

	# Janela para L–T (ida L, volta T)
	tLT_est = t0 + h/vL0 + h/vT0
	gate_LT = (tLT_est - args.margem, tLT_est + args.margem)

	# --- Picos dentro de cada gate ---
	tLL, iLL = find_peak_in_gate(t, env, *gate_LL)
	tTT, iTT = find_peak_in_gate(t, env, *gate_TT)
	tLT, iLT = find_peak_in_gate(t, env, *gate_LT)

	# --- Cálculo de velocidades (pulso–eco) ---
	VL = None if (tLL is None) else (2*h / (tLL - t0))
	VT = None if (tTT is None) else (2*h / (tTT - t0))

	# --- Checagem L–T ---
	LT_check = None
	if (tLT is not None) and (VL is not None) and (VT is not None):
		LT_check = (tLT - t0) - (h/VL + h/VT)

	# --- Rigidezes fora do plano ---
	C33 = None if (VL is None) else (rho * VL**2)
	C44 = None if (VT is None) else (rho * VT**2)

	# --- Relatório no terminal ---
	print("\n=== RESULTADOS (A-scan -> CFRP) ===")
	print(f"Espessura h = {h:.6f} m | Densidade ρ = {rho:.1f} kg/m³")
	print(f"t0 (emissão) = {t0:.8e} s")
	print(f"t_LL = {None if tLL is None else f'{tLL:.8e} s'} | V_L(z) = {None if VL is None else f'{VL:.2f} m/s'}")
	print(f"t_TT = {None if tTT is None else f'{tTT:.8e} s'} | V_T(z) = {None if VT is None else f'{VT:.2f} m/s'}")
	print(f"t_LT = {None if tLT is None else f'{tLT:.8e} s'} | Checagem LT (s) = {LT_check}")
	print(f"C33 = {None if C33 is None else f'{C33:.3e} Pa'} | C44 = {None if C44 is None else f'{C44:.3e} Pa'}")

	# --- Salva CSV com resultados ---
	out = {
		'h_m': h, 'rho_kgm3': rho,
		't0_s': t0,
		'tLL_s': tLL, 'VL_mps': VL, 'C33_Pa': C33,
		'tTT_s': tTT, 'VT_mps': VT, 'C44_Pa': C44,
		'tLT_s': tLT, 'LT_check_s': LT_check
	}
	out_df = pd.DataFrame([out])
	import tempfile
	out_csv = tempfile.NamedTemporaryFile(delete=False, suffix="_resultado.csv").name
	out_df.to_csv(out_csv, index=False)
	print(f"\nArquivo salvo: {out_csv}")

	# --- Figura do A-scan + envelope + marcas ---
	fig, ax = plt.subplots(figsize=(10,4), dpi=150)
	ax.plot(t, amp, color='steelblue', lw=0.8, label='A-scan')
	ax.plot(t, env, color='crimson', lw=1.2, label='Envelope (RMS)')
	# Marcas de picos
	for tp, lab in [(tLL,'L–L'), (tLT,'L–T'), (tTT,'T–T')]:
		if tp is not None:
			ax.axvline(tp, color='green', ls='--', lw=1)
			ax.text(tp, 0.9, lab, rotation=90, va='top', ha='center', color='green')
	# Gates
	for (g, name) in [(gate_LL,'Gate L–L'), (gate_LT,'Gate L–T'), (gate_TT,'Gate T–T')]:
		ax.axvspan(g[0], g[1], color='gray', alpha=0.1, label=name)
	ax.set_xlabel('Tempo (s)'); ax.set_ylabel('Amplitude (norm.)')
	ax.set_title('A scan e detecção automática de ecos (CFRP)')
	ax.legend(loc='upper right', ncol=2, fontsize=8)
	fig.tight_layout()
	out_png = tempfile.NamedTemporaryFile(delete=False, suffix="_ascan_resultado.png").name
	fig.savefig(out_png, dpi=150)
	print(f"Figura salva: {out_png}")

if __name__ == "__main__":
	import streamlit as st
	st.title("Processamento de A-scan CFRP")
	st.write("Preencha os parâmetros abaixo para processar o arquivo de A-scan.")

	csv_file = st.file_uploader("Selecione o arquivo CSV de A-scan", type=["csv"])
	espessura_h = st.number_input("Espessura h [m]", min_value=0.0, format="%0.6f")
	densidade = st.number_input("Densidade [kg/m³]", min_value=0.0, format="%0.2f")
	vL0 = st.number_input("Estimativa inicial V_L [m/s]", min_value=0.0, value=2900.0, format="%0.2f")
	vT0 = st.number_input("Estimativa inicial V_T [m/s]", min_value=0.0, value=1400.0, format="%0.2f")
	margem = st.number_input("Margem dos gates [s]", min_value=0.0, value=2e-6, format="%0.2e")
	win = st.number_input("Largura da janela RMS (amostras)", min_value=1, value=401, step=1)

	if st.button("Processar"):
		if csv_file is not None and espessura_h > 0 and densidade > 0:
			import tempfile
			import types
			# Salva arquivo temporário
			with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
				tmp.write(csv_file.read())
				tmp_path = tmp.name
			# Cria objeto de argumentos
			args = types.SimpleNamespace(
				csv=tmp_path,
				espessura_h=espessura_h,
				densidade=densidade,
				vL0=vL0,
				vT0=vT0,
				margem=margem,
				win=win
			)
			# Executa processamento e captura resultados
			import io
			import sys
			# Redireciona stdout para capturar prints
			old_stdout = sys.stdout
			sys.stdout = mystdout = io.StringIO()
			# Executa main(args) e obtém nomes dos arquivos gerados
			main(args)
			sys.stdout = old_stdout
			resultado = mystdout.getvalue()
			st.success("Processamento concluído!")
			st.markdown("### Resultados do processamento:")
			st.code(resultado)
			# Descobre nomes dos arquivos gerados
			import os
			# Busca nomes dos arquivos temporários gerados (últimas impressões do main)
			import re
			csv_match = re.search(r"Arquivo salvo: (.+_resultado.csv)", resultado)
			img_match = re.search(r"Figura salva: (.+_ascan_resultado.png)", resultado)
			csv_out = csv_match.group(1) if csv_match else None
			img_out = img_match.group(1) if img_match else None
			# Botão para baixar CSV
			if csv_out and os.path.exists(csv_out):
				with open(csv_out, "rb") as f:
					st.download_button("Baixar resultados (CSV)", f, file_name=os.path.basename(csv_out), mime="text/csv")
			# Exibe a figura no Streamlit
			if img_out and os.path.exists(img_out):
				import streamlit as st2
				from PIL import Image
				with open(img_out, "rb") as f:
					st.download_button("Baixar figura (PNG)", f, file_name=os.path.basename(img_out), mime="image/png")
					f.seek(0)
					image = Image.open(f)
					st.image(image, caption="A-scan e detecção automática de ecos (CFRP)", width=800)
			st.info("Os arquivos não são mais salvos na pasta do projeto, apenas para download.")
		else:
			st.error("Preencha todos os campos obrigatórios e selecione um arquivo CSV.")
