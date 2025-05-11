const HowItWorksPage = () => (
	<div className="container mx-auto p-6">
	  <h1 className="text-3xl font-bold mb-6">How It Works</h1>
	  <section className="mb-8">
		<h2 className="text-2xl font-semibold mb-4">ECG Signal Acquisition</h2>
		<p className="mb-4">
		  ECG signals are recorded using electrodes placed on the skin, capturing the heart's electrical activity over time. These signals include components like the P wave, QRS complex, and T wave, which reflect different phases of the cardiac cycle.
		</p>
	  </section>
	  <section className="mb-8">
		<h2 className="text-2xl font-semibold mb-4">Preprocessing</h2>
		<p className="mb-4">
		  Raw ECG signals contain noise and artifacts. Preprocessing involves:
		  - **Band-pass filtering (0.5–35 Hz)**: Removes high-frequency noise and low-frequency baseline wander.
		  - **Baseline wander correction**: Adjusts for slow drifts using techniques like polynomial fitting.
		</p>
	  </section>
	  <section className="mb-8">
		<h2 className="text-2xl font-semibold mb-4">Feature Extraction</h2>
		<p className="mb-4">
		  Features are derived from the preprocessed signal to characterize heartbeats:
		  - **Time-domain**: RR intervals (time between consecutive R peaks), P-QRS-T durations.
		  - **Frequency-domain**: Power spectral density via Fourier Transform, highlighting frequency components.
		  - **Time-frequency domain**: Wavelet transform, capturing both time and frequency information.
		</p>
		<p className="mb-4">
		  See <a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6513007/" target="_blank" className="text-blue-600">this paper</a> for detailed methodologies.
		</p>
	  </section>
	  <section className="mb-8">
		<h2 className="text-2xl font-semibold mb-4">Classification</h2>
		<p className="mb-4">
		  Traditional ML algorithms classify heartbeats:
		  - **SVM**: Separates classes with a hyperplane.
		  - **RF**: Combines multiple decision trees.
		  - **kNN**: Assigns class based on nearest neighbors.
		  - **GBDT**: Boosts weak learners iteratively.
		</p>
	  </section>
	  <section className="mb-8">
		<h2 className="text-2xl font-semibold mb-4">Understanding ECG Heartbeats</h2>
		<p className="mb-4">
		  An ECG measures the heart's electrical activity. Key components include:
		  - **P Wave**: Atrial depolarization.
		  - **QRS Complex**: Ventricular depolarization.
		  - **T Wave**: Ventricular repolarization.
		  Abnormalities (e.g., Premature Ventricular Contraction) indicate potential heart conditions.
		</p>
	  </section>
	</div>
);

export default HowItWorksPage;