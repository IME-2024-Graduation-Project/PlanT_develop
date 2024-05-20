import React, { useState, useEffect } from "react";
import { Link, useLocation } from 'react-router-dom';
import { Card, CardHeader, CardBody, CardTitle, Row, Col, Button, Table, UncontrolledDropdown, DropdownToggle, DropdownMenu, DropdownItem } from "reactstrap";
import MapMarker from "./Mapmarker";
import PlacesList from "variables/Places";

function TableList() {
  const location = useLocation();
  console.log(location);
  const { Tagvalue = '', City1 = '', City2 = '', move = ''} = location.state || {};
  const [visitCount, setVisitCount] = useState(parseInt(localStorage.getItem('visitCount')) || 0);

  const Places = PlacesList;

  const Places2 = Places.filter(place => place.tag === '쇼핑' || place.tag === '관광');
  // const Places3 = Places.filter(place => place.tag === '식사' || place.tag === '카페');


  const [query, setQuery] = useState("");
  const [searchedPlaces, setSearchedPlaces] = useState(Places);
  const [selectedPlaces, setSelectedPlaces] = useState([]);
  const [selectedTag, setSelectedTag] = useState(Tagvalue);
  const [selectedCity, setSelectedCity] = useState("");

const senddata2 = {
    moved: {move},
    city1: '서울', 
    city2: '부산', 
    place1: '그랜트 하얏트 서울',
    place1_city: '서울',
    place1_tag: '숙박', 
    place2: '한화리조트 해운대',
    place2_city: '부산',
    place2_tag: '숙박',}

  const senddata3 = {
    moved: {move},
    city1: '서울', 
    city2: '부산', 
    place1: '그랜트 하얏트 서울',
    place1_city: '서울',
    place1_tag: '숙박', 
    place2: '한화리조트 해운대',
    place2_city: '부산',
    place2_tag: '숙박',
    place3: '서울역',
    place3_city: '서울',
    place3_tag: '이동',}  

  const senddata4 = {
    moved: {move},
    city1: '서울', 
    city2: '부산', 
    place1: '그랜트 하얏트 서울',
    place1_city: '서울',
    place1_tag: '숙박', 
    place2: '한화리조트 해운대',
    place2_city: '부산',
    place2_tag: '숙박', 
    place3: '서울역',
    place3_city: '서울',
    place3_tag: '이동',
    place4: '부산역',
    place4_city: '부산',
    place4_tag: '이동',}

  let senddata = [];

  if (visitCount === 1) {
    senddata = senddata2;
  }
  if (visitCount === 2) {
    senddata = senddata3;
  }
  if (visitCount === 3) {
    senddata = senddata4;
  }

  const handleSearch = () => {
    let filteredPlaces = Places;
    if (!query && !selectedTag && !Tagvalue) {
      // 검색어와 선택된 태그가 모두 없는 경우 전체 도시 목록을 표시
      filteredPlaces = Places2;
    } 
    if (query) {
      filteredPlaces = filteredPlaces.filter(place => place.name.toLowerCase().includes(query.toLowerCase()));
    }

    if (selectedTag) {
      filteredPlaces = filteredPlaces.filter(place => place.tag === selectedTag);
    }

    if (selectedCity) {
      filteredPlaces = filteredPlaces.filter(place => place.city === selectedCity);
    }

    setSearchedPlaces(filteredPlaces);
  };

  const handleTagSelect = (tag) => {
    setSelectedTag(tag);
    setSelectedCity(""); // 선택된 태그가 변경되면 선택된 도시를 초기화
    const filteredPlaces = Places.filter(place => place.tag === tag);
    setSearchedPlaces(filteredPlaces);
  };

  const handleCitySelect = (city) => {
    setSelectedCity(city);
    let picktag = ''
    if (Tagvalue) {picktag = Tagvalue;}
    else {picktag = selectedTag;}
    const filteredPlaces = Places.filter(place => place.city === city && place.tag === picktag);

    setSearchedPlaces(filteredPlaces);
  };

  useEffect(() => {
    // 페이지가 처음 로드될 때 전체 도시 목록을 표시
    if (Tagvalue) {
      const filteredPlaces = Places.filter(place => place.tag === Tagvalue);
      setSearchedPlaces(filteredPlaces);
      setVisitCount(prevCount => prevCount + 1);
      if (visitCount + 1 > 3) {
        localStorage.removeItem('visitCount');
      }
      else {
      // 방문 횟수를 로컬 스토리지에 저장합니다.
      localStorage.setItem('visitCount', visitCount + 1);
      }
    }
    else {
      setSearchedPlaces(Places2);
    }
  }, []);

  const handlePlaceClick = (place) => {
    if (selectedPlaces.find(item => item.id === place.id)) {
      setSelectedPlaces(selectedPlaces.filter(item => item.id !== place.id));
    } else {
      setSelectedPlaces([...selectedPlaces, place]);
      setPlaceIds(prevPlaceIds => [...prevPlaceIds, place.id]);
    }
  };

  const [placeIds, setPlaceIds] = useState([]);



  return (
    <>
      <div className="content">
        <Row>
          <Col md="12">
            <Card className="demo-icons">
              <CardHeader>
                <CardTitle tag="h5">Input for Trip(2/3)</CardTitle>
              </CardHeader>
              <hr />
              <CardBody className="all-icons">
                <div id="icons-wrapper">
                  <section>
                    <div>
                  {!Tagvalue && (    
                    <div
                      id="map"
                      className="map"
                      style={{ position: "relative", overflow: "hidden" }}
                    >
                      <MapMarker placeIds={placeIds} />
                    <br/>
                    <hr/>
                    </div>
                  )}
                    <br/>
                      <div className='Travel_City'>
                        <input
                          type="search"
                          placeholder="방문할 장소 검색"
                          value={query}
                          onChange={(e) => setQuery(e.target.value)}
                        />
                        <button onClick={handleSearch}>검색</button>
                        <div className="text-right">
                          <UncontrolledDropdown group>
                            <DropdownToggle caret color="info">
                              Cities
                            </DropdownToggle>
                            <DropdownMenu>
                              <DropdownItem onClick={() => handleCitySelect("서울")}>서울</DropdownItem>
                              <DropdownItem onClick={() => handleCitySelect("부산")}>부산</DropdownItem>
                            </DropdownMenu>
                          </UncontrolledDropdown>
                          <UncontrolledDropdown group>
                            <DropdownToggle caret color="info">
                              Tags
                            </DropdownToggle>
                            
                            {Tagvalue && (
                            <DropdownMenu>
                              <DropdownItem onClick={() => handleTagSelect(Tagvalue)}>{Tagvalue}</DropdownItem>
                            </DropdownMenu>
                            )}
                            
                            {!Tagvalue && (
                            <DropdownMenu> 
                              <DropdownItem onClick={() => handleTagSelect("쇼핑")}>쇼핑</DropdownItem>
                              {/* <DropdownItem onClick={() => handleTagSelect("식사")}>식사</DropdownItem> */}
                              <DropdownItem onClick={() => handleTagSelect("관광")}>관광</DropdownItem>
                              {/* <DropdownItem onClick={() => handleTagSelect("카페")}>카페</DropdownItem>
                              <DropdownItem onClick={() => handleTagSelect("이동")}>이동</DropdownItem> */}
                              </DropdownMenu>
                            )}
                          </UncontrolledDropdown>
                        </div>
                      <div style={{ maxHeight: "400px", overflowY: "auto" }}>
                        <Table responsive className="text-center">
                          <thead className="text-primary">
                            <tr>
                              <th>Name</th>
                              <th>Tag</th>
                              <th>City</th>
                              <th>Action</th>
                            </tr>
                          </thead>
                          <tbody>
                            {searchedPlaces.map(place => (
                              <tr className="text-center" key={place.id}>
                                <td>{place.name}</td>
                                <td>{place.tag}</td>
                                <td>{place.city}</td>
                                <td><Button className="btn-round" color="info" onClick={() => handlePlaceClick(place)}>Add</Button></td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                      </div>
                      <br />
                      <hr />
                      <br />
                      <div>
                        <Table responsive className="text-center">
                          <thead className="text-primary">
                            <tr>
                              <th>Name</th>
                              <th>Tag</th>
                              <th>City</th>
                              <th>Action</th>
                            </tr>
                          </thead>
                          <tbody>
                            {selectedPlaces.map(place => (
                              <tr className="text-center" key={place.id}>
                                <td>{place.name}</td>
                                <td>{place.tag}</td>
                                <td>{place.city}</td>
                                <td><Button className="btn-round" color="danger" onClick={() => handlePlaceClick(place)}>Delete</Button></td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                      </div>
                      </div>
                      <br />
                      <h5> {selectedPlaces.length}/6 (최대 2곳 * 3일) </h5>
                      <br />
                      <br />
                      <hr />
                      <br />
                      <div>
                        <Col className="ml-auto mr-auto" lg="8">
                          <Row>
                            <Col md="4">
                              {!Tagvalue && (   
                                <Button
                                  block
                                  className="btn-round"
                                  color="primary"
                                  href="http://localhost:3000/admin/icons">
                                  Prev
                                </Button>
                              )}
                            </Col>
                            <Col md="4">
                            </Col>
                            <Col md="4">
                              {!Tagvalue && ( 
                                <Button
                                  block
                                  className="btn-round"
                                  color="primary"
                                  type="submit"
                                  href="http://localhost:3000/admin/notifications">
                                  Next
                                </Button>
                              )}
                              {Tagvalue && ( 
                                <Link to={"/admin/icons"} state={senddata}>
                                  <Button
                                    block
                                    className="btn-round"
                                    color="primary">
                                    Confirm
                                  </Button>
                                </Link>
                              )}
                            </Col>
                          </Row>
                        </Col>
                      </div>
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

export default TableList;
