
import './App.css'

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import InteractivePage from './InteractivePage/page';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import HowItWorksPage from './pages/HowItWorksPage';
// import LiveDemoPage from './pages/LiveDemoPage';
import ModelPage from './pages/ModelPage';
import AboutPage from './pages/AboutPage';
function App() {

  return (
    
    <Router>
    <div className="min-h-screen flex flex-col">
    <Header/>
    <main className="flex-grow">
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/how-it-works" element={<HowItWorksPage />} />
        {/* <Route path="/live-demo" element={<LiveDemoPage />} /> */}
        <Route path="/model" element={<ModelPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/live-demo" element={<InteractivePage />} />
      </Routes>
    
    </main>
    <Footer />
  </div>
  </Router>
  )
}

export default App
