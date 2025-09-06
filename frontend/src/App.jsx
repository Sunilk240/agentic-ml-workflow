// import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
// import { useState } from 'react';
// import UploadPage from './components/UploadPage';
// import ChatInterface from './components/ChatInterface';
// import './styles/globals.css';


// function App() {
//   const [sessionId, setSessionId] = useState(null);
//   const [filename, setFilename] = useState('');

//   return (
//     <Router>
//       <div className="app">
//         <Routes>
//           <Route 
//             path="/" 
//             element={
//               <UploadPage 
//                 onUploadSuccess={(sessionId, filename) => {
//                   setSessionId(sessionId);
//                   setFilename(filename);
//                 }} 
//               />
//             } 
//           />
//           <Route 
//             path="/chat" 
//             element={
//               <ChatInterface 
//                 sessionId={sessionId} 
//                 filename={filename}
//               />
//             } 
//           />
//         </Routes>
//       </div>
//     </Router>
//   );
// }

// export default App;

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import UploadPage from './components/UploadPage';
import ChatInterface from './components/ChatInterface';
import UserGuide from './components/UserGuide'; // Add this import
import './styles/globals.css';

function App() {
  const [sessionId, setSessionId] = useState(null);
  const [filename, setFilename] = useState('');

  return (
    <Router>
      <div className="app">
        <Routes>
          <Route 
            path="/" 
            element={
              <UploadPage 
                onUploadSuccess={(sessionId, filename) => {
                  setSessionId(sessionId);
                  setFilename(filename);
                }} 
              />
            } 
          />
          
          <Route 
            path="/chat" 
            element={
              <ChatInterface 
                sessionId={sessionId} 
                filename={filename} 
              />
            } 
          />
          
          {/* Add this new route for the User Guide */}
          <Route 
            path="/guide" 
            element={<UserGuide />} 
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
