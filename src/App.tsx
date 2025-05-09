
import './App.css'

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import InteractivePage from './InteractivePage/page';

function App() {

  return (
    <Router>
      <Routes>
          <Route path="/InteractivePage" element={<InteractivePage />} />
      </Routes>
    </Router>
  )
}

export default App
