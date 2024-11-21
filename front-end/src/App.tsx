import {
  Route, 
  createBrowserRouter, 
  createRoutesFromElements, 
  RouterProvider
} from 'react-router-dom'

import MainLayout from './layouts/MainLayout';
import HomePage from './pages/HomePage';
import ChartsPage from './pages/ChartsPage';
import ExtractedReqPage from './pages/ExtractedReqPage';
import NotFoundPage from './pages/NotFoundPage';

const App = () => {




  const router = createBrowserRouter(
    createRoutesFromElements(
      <Route path='/' element={<MainLayout />}>
  
        <Route index element={<HomePage />} />
        <Route path='/charts' element={<ChartsPage />} />
        <Route path='/extracted-requirements' element={<ExtractedReqPage />} />
        <Route path='*' element={<NotFoundPage />}></Route>
      </Route>
  
  )
  );


  return (
    <RouterProvider router={router} />
  )
}

export default App
