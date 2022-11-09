import React from "react";
import { Audio } from "react-loader-spinner";
const SpinnerComponent = () => {
  return (
    <div className="Loading">
      <p>Loading ...</p>
      <span>
        <Audio
          height="80"
          width="80"
          radius="9"
          color="green"
          ariaLabel="three-dots-loading"
          wrapperStyle
          wrapperClass
        />
      </span>
    </div>
  );
};

export default SpinnerComponent;
