/*!

=========================================================
* Paper Dashboard React - v1.3.2
=========================================================

* Product Page: https://www.creative-tim.com/product/paper-dashboard-react
* Copyright 2023 Creative Tim (https://www.creative-tim.com)

* Licensed under MIT (https://github.com/creativetimofficial/paper-dashboard-react/blob/main/LICENSE.md)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/
import React, { useState, useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
// import { Link } from 'react-router-dom';
// reactstrap components
import { Card, CardHeader, CardBody, CardTitle, Row, Col, Button, Table, UncontrolledDropdown, DropdownToggle, DropdownMenu, DropdownItem} from "reactstrap";
import axios from "axios";

function Icons() {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [selectedPlaces, setSelectedPlaces] = useState([]);
  // const [userinfo, setUserinfo] = useState('');
  const storedUser = localStorage.getItem("user");

  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth() + 1;
  const date = now.getDate();
  const currentTime = year + '-' + month.toString().padStart(2, '0') + '-' + date.toString().padStart(2, '0');

  // const getUserState = async () => {
  //   try {
  //     const response = await axios.post('http://localhost:8000/account/state/');
  //     // 응답 데이터에서 사용자 정보를 상태로 설정
  //     console.log(response.data);
  //     setUserinfo(response.data);
  //   } catch (error) {
  //     // 요청 실패 시 에러를 콘솔에 출력
  //     console.error("Failed to fetch user:", error);
  //   }
  // };

  // React.useEffect(() => {
  //   getUserState();
  //   console.log(userinfo)
  // }, []);

  const handleStartDateChange = (event) => {
      const selectedStartDate = event.target.value + 'T00:00:00'; // 시간 추가
      if (selectedStartDate && currentTime < selectedStartDate) {
          setStartDate(event.target.value);
      } else {
          setStartDate('');
          alert('조건에 맞게 입력해주세요.');
      }
  };

  const handleEndDateChange = (event) => {
    const selectedEndDate = event.target.value;
    if (startDate && selectedEndDate && startDate < selectedEndDate) {
      setEndDate(selectedEndDate);
    } else {
      setEndDate('');
      alert('조건에 맞게 입력해주세요.');
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
      setSubmitted(true);
  };
  
  const cities = [
    { id: 1, name: '서울', tag: 'All' },
    { id: 2, name: '부산', tag: 'All' },
    { id: 3, name: '인천', tag: '이동' },
    { id: 4, name: '수원', tag: '관광' },
    { id: 5, name: '천안', tag: '식사' },
  ];

  
  const [query, setQuery] = useState("");
  const [searchedCities, setSearchedCities] = useState([]);
  const [selectedCities, setSelectedCities] = useState([]);
  const [selectedTag, setSelectedTag] = useState("");
  

  // const handleSearch = () => {
  //   axios.post('http://127.0.0.1:8000/api/sender/', {table : 'city', filter_dict : {'city_name' : query}})
  //     .then(response => {
  //       setResult(response.data['1']);
  //       setResult2(response.data['2']); // 받은 응답에서 이름을 가져와 결과로 설정
  //     })
  //     .catch(error => {
  //       console.error('Error retrieving data:', error);
  //     });

  //   e.preventDefault();
  //   const filteredselectedCity = cities.filter(({ city }) =>
  //   city.includes(query)
  //   );
  //   setSearchedCities(filteredselectedCity);
  // }

  const handleSearch = () => {
    if (!query && !selectedTag) {
      // 검색어와 선택된 태그가 모두 없는 경우 전체 도시 목록을 표시
      setSearchedCities(cities);
    } else {
      let filteredCities = cities;
  
      if (query) {
        // 검색어가 있는 경우 해당하는 도시만 필터링
        filteredCities = filteredCities.filter(({ name }) =>
          name.toLowerCase().includes(query.toLowerCase())
        );
      }
  
      if (selectedTag) {
        // 선택된 태그가 있는 경우 해당하는 도시만 필터링
        filteredCities = filteredCities.filter(({ tag }) => tag === selectedTag);
      }
  
      setSearchedCities(filteredCities);
    }
  };

  const handleTagSelect = (tag) => {
    setSelectedTag(tag);
    // 선택된 태그에 따라 도시 목록 필터링
    const filteredCities = cities.filter(city => city.tag === tag);
    setSearchedCities(filteredCities);
  };

  useEffect(() => {
    // 페이지가 처음 로드될 때 전체 도시 목록을 표시
    setSearchedCities(cities);
  }, []);

  const handleCityClick = (city) => {
    if (selectedCities.find(item => item.id === city.id)) {
      setSelectedCities(selectedCities.filter(item => item.id !== city.id));
    } else {
      setSelectedCities([...selectedCities, city]);
    }
  };

  
  // const [transportation, setTransportation] = useState('');
  // const [departureLocation, setDepartureLocation] = useState('');
  // const [arrivalLocation, setArrivalLocation] = useState('');

  const [selectedItem, setSelectedItem] = useState('여행 중 이동 수단을 선택하세요');

  const handleItemClick = (item) => {
    setSelectedItem(item);
  };


  // const handleArrivalLocationChange = (event) => {
  //   setSelectedTag('이동');
  // };
// 선택 도시목록 + 숙박 보내기
  const handleHotelButtonClick = () => {
  //   // 선택된 도시 목록과 '숙박' 정보를 세션 스토리지에 저장
  //   sessionStorage.setItem('selectedCities', JSON.stringify(selectedCities.map(city => city.name)));


  //   // 페이지 이동
   };

  // const handleTransportationChange = (event) => {
  //   setTransportation(event.target.value);
  // };

  // const handleSubmit = (event) => {
  //   event.preventDefault();
  //   if (moveDates && departureLocation && arrivalLocation && transportation) {
  //     setSubmitted(true);
  //   } else {
  //     alert('여행 이동 날짜, 출발 지역, 도착 지역, 이동 수단을 모두 입력하세요.');
  //   }
  // };

  // 임시로 사용할 이동 수단 옵션 리스트
  // const transportationOptions = [
  //   '자차',
  //   '대중교통',
  // ];  

  const location = useLocation();
  console.log(location);
  const {
    moved='',
    city1= '', 
    city2= '', 
    place1= '',
    place1_city= '',
    place1_tag= '', 
    place2= '',
    place2_city= '',
    place2_tag='', 
    place3= '',
    place3_city= '',
    place3_tag='',
    place4= '',
    place4_city= '',
    place4_tag= ''} = location.state || {};

    useEffect(() => {
      if (city1) {
        const selectedCity1 = cities.find(city => city.name === city1);
        setSelectedCities(prevCities => [...prevCities, selectedCity1]);
      }
      if (city2) {
        const selectedCity2 = cities.find(city => city.name === city2);
        setSelectedCities(prevCities => [...prevCities, selectedCity2]);
      }
    }, [city1, city2]);
  
    useEffect(() => {
      if (moved) {
        setStartDate('2024-04-20');
        setEndDate('2024-04-23');
        setMoveDate('2024-04-22');
      }
      
    }, [moved]);

    // place1부터 place4_tag까지의 값이 존재하는 경우 선택된 장소 리스트에 반영
    useEffect(() => {
      const selectedPlaces = [];
      if (place1) {
        const place1Obj = { id: 1, city: place1_city, hotel: place1, move: place1_tag };
        selectedPlaces.push(place1Obj);
      }
      if (place2) {
        const place2Obj = { id: 2, city: place2_city, hotel: place2, move: place2_tag };
        selectedPlaces.push(place2Obj);
      }
      if (place3) {
        const place3Obj = { id: 3, city: place3_city, hotel: place3, move: place3_tag };
        selectedPlaces.push(place3Obj);
      }
      if (place4) {
        const place4Obj = { id: 4, city: place4_city, hotel: place4, move: place4_tag };
        selectedPlaces.push(place4Obj);
      }
      setSelectedPlaces(selectedPlaces);
    }, [place1, place1_city, place1_tag, place2, place2_city, place2_tag, place3, place3_city, place3_tag, place4, place4_city, place4_tag]);

    const [moveDate, setMoveDate] = useState({moved});

    const handleMoveDateChange = (event) => {
      const mdate = event.target.value
      if (mdate && startDate && endDate && startDate <= mdate && mdate <= endDate) {
        setMoveDate(mdate);
      } 
      else {
        alert('이동 날짜는 여행 시작일 이후이고 종료일 이전이어야 합니다.');
      }
    };

    let datahotel = { Tagvalue: '숙박', City1: '서울', City2: '부산', move: {moveDate} };
    let datamove = { Tagvalue: '이동', City1: '서울', City2: '부산', move: {moveDate} };


  return (
    <>
      <div className="content">
        <Row>
          <Col md="12">
            <Card className="demo-icons">
              <CardHeader>
                <CardTitle tag="h5">Input for Trip(1/3) {storedUser}</CardTitle>
                {/* <p className="card-category">
                  Handcrafted by our friends from{" "}
                  <a href="https://nucleoapp.com/?ref=1712">NucleoApp</a>
                </p> */}
              </CardHeader>
              <hr/>
              <CardBody className="all-icons">
                <br/>
                <div id="icons-wrapper">
                  <section>
                    <div>
                      {/* <h1>여행 시작일과 종료일 입력</h1> */}
                      {/* <form onSubmit={handleSubmit}> */}
                        <Col className="ml-auto mr-auto" lg="8">
                          <div>
                            <label htmlFor="startDate">start date:</label>
                            <input
                              type="date"
                              id="startDate"
                              value={startDate}
                              onChange={handleStartDateChange}
                            />
                          {/* </div>
                          <div> */}
                            <label htmlFor="endDate">end date:</label>
                            <input
                              type="date"
                              id="endDate"
                              value={endDate}
                              onChange={handleEndDateChange}
                            />
                          </div>
                          <br/>
                          <hr/>
                          <br/>
                          <div className='Travel_City'>
                            <input
                              type="search"
                              placeholder="여행할 지역 검색"
                              value={query}
                              onChange={(e) => setQuery(e.target.value)}
                            />
                            <button onClick={handleSearch}>검색</button>
                          </div>  
                            <div className="text-right">
                            <UncontrolledDropdown group>
                                <DropdownToggle caret color="info">
                                    Tags
                                </DropdownToggle>
                                <DropdownMenu>
                                  <DropdownItem onClick={() => handleTagSelect("All")}>All</DropdownItem>
                                  <DropdownItem onClick={() => handleTagSelect("쇼핑")}>쇼핑</DropdownItem>
                                  <DropdownItem onClick={() => handleTagSelect("식사")}>식사</DropdownItem>
                                  <DropdownItem onClick={() => handleTagSelect("관광")}>관광</DropdownItem>
                                  <DropdownItem onClick={() => handleTagSelect("카페")}>카페</DropdownItem>
                                  <DropdownItem onClick={() => handleTagSelect("이동")}>이동</DropdownItem>
                                </DropdownMenu>
                            </UncontrolledDropdown>
                            </div>
                            {/* <Table responsive>
                              <thead>
                                <tr className="text-center">
                                  <th>Name</th>
                                  <th>Tag</th>
                                  <th>Action</th>
                                </tr>
                              </thead>
                              <tbody>
                              
                                {searchedCities.map(city => (
                                  <tr className="text-center">
                                    <td>{city.name}</td>
                                    <td>{city.tag}</td>
                                    <td><Button className="btn-round" color="info" key={city.id} onClick={() => handleCityClick(city)}>Add</Button></td>
                                  </tr>
                                ))}
                              </tbody>
                            </Table> */}
                            <br/>
                           
                            <h5>Selected cities</h5>
                            <Table responsive>
                              <thead>
                                <tr className="text-center">
                                  <th>Name</th>
                                  <th>Tag</th>
                                  <th>Action</th>
                                </tr>
                              </thead>
                              <tbody>
                                {selectedCities.length > 0 && selectedCities.map(city => (
                                  <tr className="text-center" key={city.id}>
                                    <td>{city.name}</td>
                                    <td>{city.tag}</td>
                                    <td>
                                      <Button
                                        className="btn-round"
                                        color="danger"
                                        onClick={() => handleCityClick(city)}
                                        style={{ cursor: 'pointer' }}
                                      >
                                        Delete
                                      </Button>
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </Table>
        
                          {/* <div className='Searched_selectedCity'>
                            <header>총 {searchedCities.length}개의 도시가 검색되었습니다.</header>
                            <ul>
                              {searchedCities.map(cities => (
                                <li key={cities.code}>
                                  {cities.city}
                                  <button onClick={() => handleSelect(cities.city)}>선택</button>
                                </li>
                              ))}
                            </ul>
                            <p>선택된 지역 목록</p>
                            <ul>
                              {selectedCities.map((cities, index) => (
                                <li key={index}>{cities.city}</li>
                              ))}
                            </ul>
                          </div> */}
                          <br/>
                          <hr/>
                          <br/>
                          <div>
                            {/* <h6>여행 이동 날짜 입력</h6> */}
                            {/* <form onSubmit={handleSubmit}> */}
                            <p></p>
                            <div>
                              
                              
                                <label>숙박 장소:</label>
                                <Link to={"/admin/tables"} state={datahotel}>
                                  <button>검색</button>
                                </Link>
                                <label>출발 장소:</label>
                                <Link to={"/admin/tables"} state={datamove}>
                                  <button>검색</button>
                                </Link>
                                <label>도착 장소:</label>
                                <Link to={"/admin/tables"} state={datamove}>
                                  <button>검색</button>
                                </Link>

                              <label htmlFor="moveDates">이동 날짜:</label>
                                <input
                                  type="date"
                                  id="moveDates"
                                  value={moveDate}
                                  onChange={handleMoveDateChange}
                                  
                                />
                              </div>
                              <br/>
                              <Table responsive>
                                <thead>
                                  <tr className="text-center">
                                    <th>City</th>
                                    <th>Hotel</th>
                                    <th>Move</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {selectedPlaces.length > 0 && selectedPlaces.map(place => (
                                    <tr className="text-center" key={place.id}>
                                      <td>{place.city}</td>
                                      <td>{place.hotel}</td>
                                      <td>{place.move}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </Table>
                              <br/>
                              
                              <div>
                              <UncontrolledDropdown group>
                                <DropdownToggle caret color="info">
                                  {selectedItem}
                                </DropdownToggle>
                                <DropdownMenu>
                                  <DropdownItem onClick={() => handleItemClick('자가용')}>자가용</DropdownItem>
                                  <DropdownItem onClick={() => handleItemClick('대중교통(에코 포인트 +10%)')}>
                                    대중교통(에코 포인트 +10%)
                                  </DropdownItem>
                                </DropdownMenu>
                              </UncontrolledDropdown>
                              </div>
                              {/* <button type="submit">확인</button> */}
                            {/* </form> */}
                            {/* {submitted && moveDates && departureLocation && arrivalLocation && transportation && (
                              <div>
                                <p>이동 날짜: {moveDates} 이동 수단: {transportation}</p>
                                <p>출발 지역: {departureLocation}</p>
                                <p>도착 지역: {arrivalLocation}</p>
                                <p>이동 수단: {transportation}</p>
                              </div>
                            )} */}
                          </div>
                        </Col>
                        <br/>
                        <br/>
                        <hr/>
                        <br/>
                        <Col className="ml-auto mr-auto" lg="8">
                          <Row>
                            <Col md="4">
                              <Button 
                                block
                                className="btn-round" 
                                color="primary"
                                href="http://localhost:3000/admin/icons">
                                Prev
                              </Button>
                            </Col>
                            <Col md="4">
                            </Col>
                            <Col md="4">
                              <Button 
                                block
                                className="btn-round" 
                                color="primary" 
                                type="submit"
                                href="http://localhost:3000/admin/tables">
                                Next
                              </Button>
                            </Col>
                          </Row>
                        </Col>  
      
                      {/* </form>
                      {submitted && startDate && endDate && (
                        <p>여행 시작일: {startDate}, 여행 종료일: {endDate}</p>
                      )} */}
                    </div>
                  </section>
                </div>
              </CardBody>
            </Card>
          </Col>
        </Row>
      </div>
    </>
  );
}

export default Icons;
