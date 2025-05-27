
import './App.css'

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import InteractivePage from './InteractivePage/page';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import HowItWorksEnhanced from './pages/HowItWorksEnhanced';
// import LiveDemoPage from './pages/LiveDemoPage';
import AboutPage from './pages/AboutPage';
import ShowCaseEnhanced from './pages/ShowCaseEnhanced';
import ArduinoPage from './pages/ArduinoPage';
function App() {

  return (
    
    <Router>
    {/* manually min-w-300, I don't know how to fix :( */}
    <div className="min-h-screen flex flex-col min-w-300">
    <Header/>
    {/* flex-grow idk how I work :D */}
    <main className="flex-grow">
      {/* ---------------------------------------------------------------------------------------------------------------------------------------------- */}
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/how-it-works" element={<HowItWorksEnhanced />} />
        {/* <Route path="/live-demo" element={<LiveDemoPage />} /> */}
        <Route path="/model" element={<ShowCaseEnhanced />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/live-demo" element={<InteractivePage />} />
        <Route path="/arduino" element={<ArduinoPage />} />
      </Routes>
    
    </main>
    <Footer />
  </div>
  </Router>
  )
}

export default App
