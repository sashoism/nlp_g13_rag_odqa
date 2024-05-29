import React from 'react';
import Result from './Result/Result';

import "./Results.css";

export default function Results(props) {

  console.log("props", props)

  let results = props.documents.text.map((result, index) => {
    return <Result 
        key={index} 
        document={result}
      />;
  });

  let beginDocNumber = Math.min(props.skip + 1, props.count);
  let endDocNumber = Math.min(props.skip + props.top, props.count);

  return (
    <div>
      <p className="results-info">Showing results</p>
      <div className="row row-cols-md-5 results">
        {results}
      </div>
    </div>
  );
};
