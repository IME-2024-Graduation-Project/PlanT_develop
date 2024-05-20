// Context 생성
import React, { createContext, useState } from 'react';
import Icons from 'views/Icons';
import TableList from 'views/Tables';

const InputContext = createContext();

const InputProvider = () => {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [moveDate, setMoveDate] = useState('');
  const [selectedCities, setSelectedCities] = useState([]);
  const [selectedPlaces, setSelectedPlaces] = useState([]);
  const [filterTag, setFilterTag] = useState('');

  return (
    <InputContext.Provider value={{ startDate, setStartDate, endDate, setEndDate, selectedCities, setSelectedCities, selectedPlaces, setSelectedPlaces, moveDate, setMoveDate, filterTag, setFilterTag }}>
      <Icons />
      <TableList />
    </InputContext.Provider>
  );
};

export default InputProvider;
