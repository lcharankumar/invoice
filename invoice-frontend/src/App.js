import logo from "./logo.svg";
import "./App.css";
import "./style/style.scss";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import HomeComponent from "./components/HomeComponent";
import ProgressComponent from "./components/ProgressComponent";
import TemplateComponent from "./components/TemplateComponent";
import ImageCropper from "./components/ImageCropperComponent";
import Nav_testingComponent from "./components/Nav_testingComponent";
import RectangleComponent from "./components/RectangleComponent";
import { useState, useEffect } from "react";
import ZoomComponent from "./components/ZoomComponent";
import TableComponent from "./components/TableComponent";
function App() {
  const [coordinates, setCoordinates] = useState({
    x: 20,
    y: 30,
    w: 100,
    h: 50,
  });
  const [img, setImg] = useState(null);
  useEffect(() => {
    console.log(coordinates);
    console.log(img);
  }, [coordinates]);

  return (
    <>
      <Router>
        <Routes>
          <Route path="/" element={<HomeComponent />} />
          <Route
            path="/invoice"
            element={
              <ProgressComponent
                coordinates={coordinates}
                setCoordinates={setCoordinates}
                img={img}
                setImg={setImg}
              />
            }
          />

          <Route
            path="/img"
            element={<ImageCropper setCoordinates={setCoordinates} />}
          />
          <Route
            path="/rect"
            element={<RectangleComponent coordinates={coordinates} />}
          />
          <Route
            path="/zoom"
            element={
              <ZoomComponent/>
            }
          />
        </Routes>
      </Router>
    </>
  );
}

export default App;
