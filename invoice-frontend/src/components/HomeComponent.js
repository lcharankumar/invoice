import React from "react";
import { Link, useNavigate } from "react-router-dom";
const HomeComponent = () => {
  return (
    <>
      <div className="Main">
       
        <Link to="invoice">
          <button className="Homebtn">
            {" "}
            <span> Get Started </span>
          </button>
        </Link>
      </div>
    </>
  );
};

export default HomeComponent;
