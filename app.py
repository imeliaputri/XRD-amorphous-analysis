import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import xml.etree.ElementTree as ET
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline, CubicSpline, interp1d
from scipy.ndimage import binary_dilation
import io
import warnings

warnings.filterwarnings('ignore')

# Compatibility for NumPy 1.x vs 2.x
trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# =========================================================
# 1. RIETVELD BACKGROUND CORRECTOR
# =========================================================
class RietveldBackgroundCorrector:
    def __init__(self):
        self.rietveld_bg_x = None
        self.rietveld_bg_y = None
        self.interpolated_bg = None

    def load_cif_or_txt(self, path, txt_col_idx=None):
        try:
            if path.endswith('.cif'):
                return self._load_cif(path)
            else:
                return self._load_txt(path, txt_col_idx)
        except Exception as e:
            st.error(f"Error loading background: {e}")
            return False

    def _load_cif(self, path):
        try:
            import gemmi
            doc = gemmi.cif.read_file(path)
            block = doc.sole_block()
            for loop in block.loops:
                tags = loop.tags
                if '_pd_proc_2theta_corrected' in tags:
                    idx_x = tags.index('_pd_proc_2theta_corrected')
                    bg_candidates = ['_pd_proc_intensity_bkg_calc', '_pd_proc_intensity_bkg_total',
                                     '_pd_calc_intensity_total', '_pd_proc_intensity_total']
                    idx_y = None
                    for cand in bg_candidates:
                        if cand in tags:
                            idx_y = tags.index(cand)
                            break
                    if idx_y is None:
                        continue
                    rows = [[float(v) for v in r] for r in loop if all(v.replace('.','').replace('e','').replace('-','').replace('+','').isnumeric() for v in r)]
                    data = np.array(rows)
                    self.rietveld_bg_x = data[:, idx_x]
                    self.rietveld_bg_y = data[:, idx_y]
                    return True
        except Exception:
            pass
        # Fallback: manual regex parsing for CIF
        with open(path, 'r') as f:
            content = f.read()
        import re
        loop_pattern = r'loop_\s*\n((?:\s*_.+\s*\n)+)((?:\s*[0-9.Ee+-]+\s+)+)'
        for tags_block, data_block in re.findall(loop_pattern, content, re.MULTILINE):
            tags = [t.strip() for t in re.findall(r'_(?:[^\s]+)', tags_block)]
            if '_pd_proc_2theta_corrected' not in tags:
                continue
            idx_x = tags.index('_pd_proc_2theta_corrected')
            bg_candidates = ['_pd_proc_intensity_bkg_calc', '_pd_proc_intensity_bkg_total',
                             '_pd_calc_intensity_total', '_pd_proc_intensity_total']
            idx_y = next((tags.index(c) for c in bg_candidates if c in tags), None)
            if idx_y is None:
                continue
            numbers = re.findall(r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?', data_block)
            if len(numbers) % len(tags) == 0:
                data = np.array(numbers, dtype=float).reshape(-1, len(tags))
                self.rietveld_bg_x = data[:, idx_x]
                self.rietveld_bg_y = data[:, idx_y]
                return True
        st.error("Parsing CIF gagal: tidak menemukan loop yang sesuai.")
        return False

    def _load_txt(self, path, col_idx=None):
        data = np.loadtxt(path)
        if data.ndim == 1:
            self.rietveld_bg_x = np.arange(len(data))
            self.rietveld_bg_y = data
        elif data.shape[1] == 2:
            self.rietveld_bg_x = data[:, 0]
            self.rietveld_bg_y = data[:, 1]
        else:
            if col_idx is None:
                st.error("File TXT memiliki >2 kolom. Silakan pilih indeks kolom background di sidebar.")
                return False
            self.rietveld_bg_x = data[:, 0]
            self.rietveld_bg_y = data[:, col_idx]
        return True

    def interpolate_to_grid(self, experimental_2theta, method='cubic'):
        if self.rietveld_bg_x is None or self.rietveld_bg_y is None:
            raise ValueError("No Rietveld background data loaded.")
        try:
            if method == 'cubic':
                interp = CubicSpline(self.rietveld_bg_x, self.rietveld_bg_y, extrapolate=True)
            elif method == 'linear':
                interp = interp1d(self.rietveld_bg_x, self.rietveld_bg_y, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
            else:
                interp = UnivariateSpline(self.rietveld_bg_x, self.rietveld_bg_y, s=len(self.rietveld_bg_x)*10)
            bg = interp(experimental_2theta)
            self.interpolated_bg = np.maximum(bg, 0)
            return self.interpolated_bg
        except Exception:
            interp = interp1d(self.rietveld_bg_x, self.rietveld_bg_y, kind='linear',
                              bounds_error=False, fill_value=(self.rietveld_bg_y[0], self.rietveld_bg_y[-1]))
            self.interpolated_bg = np.maximum(interp(experimental_2theta), 0)
            return self.interpolated_bg

# =========================================================
# 2. ADVANCED BACKGROUND EXTRACTOR (Iterative + Correlation)
# =========================================================
class PureBackgroundExtractor:
    def __init__(self):
        self.x = None
        self.y = None
        self.corrected_data = None
        self.background = None
        self.peak_mask = None
        self.rietveld_corrector = RietveldBackgroundCorrector()

    def parse_xrdml_bytes(self, xml_bytes):
        try:
            xml_content = xml_bytes.decode('utf-8')
            root = ET.fromstring(xml_content)
            
            def local_tag(elem):
                return elem.tag.split('}')[-1].lower()
            
            counts_elem = None
            for elem in root.iter():
                if local_tag(elem) in ['counts', 'intensities'] and elem.text:
                    counts_elem = elem
                    break
            if counts_elem is None:
                raise ValueError("Data intensitas tidak ditemukan dalam XRDML")
            
            y = np.fromstring(counts_elem.text.strip(), sep=' ').astype(float)
            
            start, end = None, None
            for elem in root.iter():
                tag = local_tag(elem)
                if tag == 'positions' and elem.get('axis', '').lower() in ['2theta', '2θ']:
                    for child in elem:
                        ct = local_tag(child)
                        if ct == 'startposition': start = float(child.text)
                        elif ct == 'endposition': end = float(child.text)
                    break
            if start is None or end is None:
                for elem in root.iter():
                    tag = local_tag(elem)
                    if tag in ['startposition', 'start']: start = float(elem.text)
                    elif tag in ['endposition', 'end']: end = float(elem.text)
                if start is None or end is None:
                    raise ValueError("Range 2theta tidak ditemukan")
                    
            self.x = np.linspace(start, end, len(y))
            self.y = y
            self.corrected_data = self.y.copy()
            return True
        except Exception as e:
            st.error(f"Gagal parsing XRDML: {e}")
            return False

    def apply_rietveld_correction(self, bg_interp):
        if self.y is None:
            return
        sigma = np.sqrt(np.maximum(self.y, 1))
        diff = self.y - bg_interp
        self.corrected_data = np.where(np.abs(diff) < 3 * sigma, 0, diff)
        self.corrected_data = np.maximum(self.corrected_data, 0.01 * self.y)

    def auto_detect_peaks(self, height_factor=0.05, prominence_factor=0.03, distance=5):
        data = self.corrected_data if self.corrected_data is not None else self.y
        peaks, _ = find_peaks(
            data,
            height=np.median(data) + height_factor * (np.max(data) - np.median(data)),
            prominence=prominence_factor * (np.max(data) - np.median(data)),
            distance=distance
        )
        return peaks

    def detect_peak_regions_correlation(self, data=None, window_size=15, correlation_threshold=0.65):
        if data is None:
            data = self.corrected_data if self.corrected_data is not None else self.y
        window_size = int(window_size)
        x_template = np.linspace(-3, 3, window_size)
        templates = [np.exp(-x_template**2 / 0.5), np.exp(-x_template**2 / 1.0), np.exp(-x_template**2 / 2.0)]
        scores = np.zeros(len(data))
        for i in range(window_size//2, len(data) - window_size//2):
            win = data[i-window_size//2:i+window_size//2+1]
            if np.std(win) > 0.1 * np.mean(win):
                max_c = max((np.corrcoef(win, t)[0,1] for t in templates if not np.isnan(np.corrcoef(win, t)[0,1])), default=0)
                scores[i] = max_c
        try:
            scores = savgol_filter(scores, min(11, len(scores)//10), 2)
        except: pass
        threshold = correlation_threshold * np.max(scores) if np.max(scores) > 0 else correlation_threshold
        mask = scores > threshold
        try:
            mask = binary_dilation(mask, structure=np.ones(7))
        except: pass
        return mask

    def conservative_snip(self, iterations=100, width_ratio=0.025):
        data = self.corrected_data if self.corrected_data is not None else self.y
        bg = data.copy()
        width = max(5, int(width_ratio * len(bg)))
        for i in range(iterations):
            for j in range(width, len(bg)-width):
                bg[j] = min(bg[j], 0.4*bg[j-width] + 0.4*bg[j+width] + 0.2*bg[j])
            if i > 10 and i % 10 == 0 and np.max(np.abs(bg - data)) < 0.01 * np.max(data):
                break
        try:
            bg = savgol_filter(bg, max(11, width*2), 2)
        except: pass
        return bg

    def iterative_refinement(self, max_iter=15, corr_target=0.75):
        data = self.corrected_data if self.corrected_data is not None else self.y
        bg = self.conservative_snip()
        auto_peaks = self.auto_detect_peaks()
        peak_mask = np.zeros_like(data, dtype=bool)
        for p in auto_peaks:
            peak_mask[max(0, p-8):min(len(data), p+8)] = True
        self.peak_mask = peak_mask | self.detect_peak_regions_correlation(data)[0]
        
        best_bg = bg.copy()
        best_corr = 0
        for it in range(max_iter):
            non_peak = ~self.peak_mask
            if np.sum(non_peak) > 50:
                w = 1.0 / np.sqrt(np.maximum(data[non_peak], 1))
                try:
                    spl = UnivariateSpline(self.x[non_peak], data[non_peak], w=w, s=len(data)*10)
                    new_bg = spl(self.x)
                    bg = 0.8 * bg + 0.2 * new_bg
                except: pass
            try:
                bg = savgol_filter(bg, min(21, len(bg)//10), 3)
            except: pass
            if np.sum(non_peak) > 10:
                try:
                    corr = np.corrcoef(data[non_peak], bg[non_peak])[0,1]
                    corr = 0 if np.isnan(corr) else corr
                except: corr = 0
            else: corr = 0
            if corr > best_corr:
                best_corr = corr
                best_bg = bg.copy()
            if corr >= corr_target: break
            auto_peaks = self.auto_detect_peaks()
            peak_mask = np.zeros_like(data, dtype=bool)
            for p in auto_peaks:
                peak_mask[max(0, p-6):min(len(data), p+6)] = True
            self.peak_mask = peak_mask | self.detect_peak_regions_correlation(data - bg)[0]
        self.background = best_bg
        return best_bg, best_corr

    def calculate_amorphous_content(self):
        data = self.corrected_data if self.corrected_data is not None else self.y
        if self.background is None or self.x is None:
            return 0, 0
        total = trapz(data, self.x)
        bg_area = trapz(self.background, self.x)
        frac = bg_area / total if total > 0 else 0
        non_peak = ~self.peak_mask if self.peak_mask is not None else np.ones_like(data, dtype=bool)
        try:
            corr = np.corrcoef(data[non_peak], self.background[non_peak])[0,1]
            corr = 0 if np.isnan(corr) else corr
        except: corr = 0
        unc = (1 - corr) * frac * 100 if corr > 0 else 5.0
        return frac * 100, max(unc, 2.0)

# =========================================================
# 3. STREAMLIT UI
# =========================================================
st.set_page_config(layout="wide", page_title="Amorphous Content Validator (Advanced)")
page_bg = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1532187643603-ba119ca4109e");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}

[data-testid="stHeader"]{
background: rgba(0,0,0,0);
}

[data-testid="stSidebar"]{
background: rgba(20,20,20,0.85);
}
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)
# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="XRD Amorphous Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<h1 style='text-align:center; color:#00B4D8;'>
🔬 XRD Amorphous Analysis Platform
</h1>

<p style='text-align:center; font-size:18px;'>
Web-Based Quantitative Analysis of Amorphous Phase Using XRD Data
</p>

<hr>
""", unsafe_allow_html=True)


xrd_file = st.file_uploader("📂 Upload File XRDML", type=['xrdml'])
bg_file = st.file_uploader("📐 Upload Background Rietveld (CIF/TXT) - Opsional", type=['cif', 'txt'])

if xrd_file:
    extractor = PureBackgroundExtractor()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xrdml') as tmp:
        tmp.write(xrd_file.read())
        tmp_path = tmp.name
    if not extractor.parse_xrdml_bytes(open(tmp_path, 'rb').read()):
        st.stop()
    os.unlink(tmp_path)

    st.sidebar.header("📊 Rentang Analisis")
    min2t = st.sidebar.slider("2θ Minimum", float(extractor.x[0]), float(extractor.x[-1]), float(extractor.x[0]), step=0.1)
    max2t = st.sidebar.slider("2θ Maksimum", min2t, float(extractor.x[-1]), float(extractor.x[-1]), step=0.1)
    mask = (extractor.x >= min2t) & (extractor.x <= max2t)
    extractor.x = extractor.x[mask]
    extractor.y = extractor.y[mask]
    extractor.corrected_data = extractor.y.copy()

    st.sidebar.header("⚙️ Parameter Rietveld")
    apply_rietveld = st.sidebar.checkbox("Kurangi Background Rietveld", value=bg_file is not None)
    rietveld_method = st.sidebar.selectbox("Metode Interpolasi", ['cubic', 'linear', 'spline'], index=0)
    
    txt_col = None
    if apply_rietveld and bg_file and bg_file.name.endswith('.txt'):
        try:
            temp = np.loadtxt(io.BytesIO(bg_file.getvalue()))
            if temp.ndim > 1 and temp.shape[1] > 2:
                txt_col = st.sidebar.number_input("Indeks Kolom BG (0-based)", 0, temp.shape[1]-1, temp.shape[1]-1)
        except: pass

    if apply_rietveld and bg_file:
        rietveld = RietveldBackgroundCorrector()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(bg_file.name)[1]) as tmp:
            tmp.write(bg_file.getvalue())
            tmp_path_bg = tmp.name
        if rietveld.load_cif_or_txt(tmp_path_bg, txt_col):
            bg_interp = rietveld.interpolate_to_grid(extractor.x, method=rietveld_method)
            extractor.apply_rietveld_correction(bg_interp)
        os.unlink(tmp_path_bg)

    st.sidebar.header("🔄 Parameter Iterasi")
    max_iter = st.sidebar.slider("Maksimum Iterasi Refinement", 5, 30, 15, step=5)
    corr_target = st.sidebar.slider("Target Korelasi", 0.5, 0.95, 0.75, step=0.05)
    
    with st.spinner("🔍 Mengekstrak background & menghitung kandungan amorf..."):
        background, correlation = extractor.iterative_refinement(max_iter=max_iter, corr_target=corr_target)
        amorphous, uncertainty = extractor.calculate_amorphous_content()

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes[0,0].plot(extractor.x, extractor.y, 'k-', lw=1, alpha=0.7, label='Original')
    if extractor.corrected_data is not None:
        axes[0,0].plot(extractor.x, extractor.corrected_data, 'b-', lw=1.2, alpha=0.8, label='Rietveld Corrected')
    axes[0,0].plot(extractor.x, extractor.background, 'r-', lw=2, label='Amorphous BG')
    axes[0,0].set_xlabel('2θ (°)'); axes[0,0].set_ylabel('Intensity')
    axes[0,0].set_title('Background Extraction Results'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

    if apply_rietveld and bg_file and hasattr(rietveld, 'interpolated_bg') and rietveld.interpolated_bg is not None:
        axes[0,1].plot(rietveld.rietveld_bg_x, rietveld.rietveld_bg_y, 'ro-', markersize=3, label='Original Rietveld')
        axes[0,1].plot(extractor.x, rietveld.interpolated_bg, 'b-', lw=2, label='Interpolated')
    else:
        axes[0,1].text(0.5, 0.5, 'Tidak ada background Rietveld', transform=axes[0,1].transAxes, ha='center', va='center', fontsize=12)
        axes[0,1].axis('off')
    axes[0,1].set_xlabel('2θ (°)'); axes[0,1].set_ylabel('Intensity')
    axes[0,1].set_title('Rietveld Interpolation'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

    residual = (extractor.corrected_data if extractor.corrected_data is not None else extractor.y) - extractor.background
    axes[1,0].plot(extractor.x, residual, 'g-', lw=1)
    axes[1,0].axhline(0, color='r', ls='--', alpha=0.5)
    axes[1,0].set_xlabel('2θ (°)'); axes[1,0].set_ylabel('Residual')
    axes[1,0].set_title('Residual Analysis'); axes[1,0].grid(True, alpha=0.3)

    summary = (f"COMPREHENSIVE AMORPHOUS ANALYSIS\n{'='*35}\n\n"
               f"Amorphous Content: {amorphous:.1f} ± {uncertainty:.1f}%\n\n"
               f"Data Status:\n"
               f"- Rietveld: {'Applied' if apply_rietveld else 'Not Applied'}\n"
               f"- Points: {len(extractor.x)} | Range: {extractor.x[0]:.1f}° - {extractor.x[-1]:.1f}°\n\n"
               f"Quality:\n"
               f"- Background Correlation: {correlation:.3f}\n"
               f"- Refinement Iterations: {max_iter}\n")
    axes[1,1].text(0.05, 0.95, summary, transform=axes[1,1].transAxes, fontsize=10, va='top', family='monospace')
    axes[1,1].set_title('Analysis Summary'); axes[1,1].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    # Export CSV
    csv_buf = io.StringIO()
    out_data = extractor.corrected_data if extractor.corrected_data is not None else extractor.y
    np.savetxt(csv_buf, np.column_stack((extractor.x, extractor.y, out_data, extractor.background, residual)),
               delimiter=',', header='2theta,Original,Corrected,Amorphous_BG,Residual', comments='')
    st.download_button("📥 Download CSV", csv_buf.getvalue(), "xrd_analysis.csv", "text/csv")
else:
    st.info("📂 Upload file XRDML untuk memulai analisis.")
    st.markdown("""
    ### 💡 Cara Pakai:
    1. Upload file `.xrdml` dari alat XRD Anda
    2. (Opsional) Upload file background Rietveld `.cif` atau `.txt`
    3. Atur rentang `2θ` dan parameter iterasi di sidebar
    4. Hasil perhitungan amorf + plot validasi akan muncul otomatis
    """)

st.markdown("""
---
© 2026 Imelia Putri Salsabila  
Web-based XRD Amorphous Analysis System
""")

