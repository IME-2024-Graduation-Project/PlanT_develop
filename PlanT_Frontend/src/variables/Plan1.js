import { useState, useEffect } from 'react';
import { Card, CardHeader, CardBody, CardTitle, Row, Col, Table, Button, Nav, NavItem, NavLink, DropdownToggle, DropdownMenu, DropdownItem, UncontrolledDropdown } from 'reactstrap';
import { Link } from 'react-router-dom';
import logo from '../plant.png';
import PlacesList from "./Places";

const Plan1 = () => {
  // 상태 설정
  const [placeInfoList, setPlaceInfoList] = useState([]);
  const [Tablevisible, setTablevisible] = useState(false);

  // useEffect를 사용하여 초기 데이터 설정 및 Tablevisible 상태에 따른 장소 정보 업데이트
  useEffect(() => {
    const placelist = [276, 277, 279, 280, 120, 122, 128, 129];
    const ecolist = [276, 277, 279, 280];

    const infolist = placelist.map(placeId => {
      const placeInfo = PlacesList.find(place => place.id === placeId);
      return {
        id: placeId,
        name: placeInfo.name,
        city: placeInfo.city,
        tag: placeInfo.tag
      };
    });

    setPlaceInfoList(infolist);
  }, []); // 빈 배열을 넣어 한 번만 실행되도록 설정

  // 버튼 클릭 시 호출되는 함수
  const handleMealSelect = () => {
    // Tablevisible 상태를 true로 업데이트하여 테이블을 렌더링하도록 설정
    setTablevisible(true);
  };

  const [selectedItem, setSelectedItem] = useState('Tags');

  const handleItemClick = (item) => {
    setSelectedItem(item);
  };

}

  return (
    <div className="content">
        <NotificationAlert ref={notificationAlert} />
        <Row>
          <Col md="12">
            <Card>
              <CardHeader>
        <CardTitle tag="h5">Input for Trip(3/3)</CardTitle>
        <Nav tabs>
          <NavItem>
            <NavLink href="#" active>1일차</NavLink>
          </NavItem>
          <NavItem>
            <NavLink href="#">2일차</NavLink>
          </NavItem>
          <NavItem>
            <NavLink disabled href="#">3일차(이동일정)</NavLink>
          </NavItem>
          <NavItem>
            <NavLink href="#">4일차</NavLink>
          </NavItem>
        </Nav>
      </CardHeader>
      <CardBody>
        <Col md="6">
          <Card className="card-plain">
            <CardHeader>
              <CardTitle tag="h5">Additional Schedule</CardTitle>
            </CardHeader>
            <CardBody>
              <UncontrolledDropdown group>
                <DropdownToggle caret color="info">
                  Tags
                </DropdownToggle>
                <DropdownMenu>
                  <DropdownItem>쇼핑</DropdownItem>
                  <DropdownItem  onClick={() => handleItemClick('식사')}>식사</DropdownItem>
                  <DropdownItem>관광</DropdownItem>
                  <DropdownItem>카페</DropdownItem>
                </DropdownMenu>
              </UncontrolledDropdown>
              <br/>
              <hr/>
              <div className="text-center">
                <Button color="warning" fade={false}>
                  <span>신라스테이 삼성(숙박)</span>
                </Button>
                <br/>
                <Button color="secondary" onClick={() => handleMealSelect()}>+</Button>
                <br/>
                <Button color="success" fade={false}>
                  <span>코엑스(관광)</span>
                </Button>
                <br/>
                <Button color="secondary">+</Button>
                <br/>
                <Button
                  color="danger"
                  fade={false}
                >
                  <span>가로수길(관광)</span>
                </Button>
                <br/>
                <Button color="secondary">+</Button>
                <br/>
                <Button
                  color="warning"
                  fade={false}
                >
                  <span>신라스테이(숙박)</span>
                </Button>
              </div>
            </CardBody>
          </Card>
        </Col>
        <Col md="6">
          <Card className="card-plain">
            <CardHeader>
              <CardTitle tag="h5">Possible Places</CardTitle>
            </CardHeader>
            <CardBody>
              <br/>
              <br/>
              <Table responsive>
                <thead className="text-primary">
                  <tr>
                    <th>Name</th>
                    <th>Tag</th>
                    <th>City</th>
                    <th>Eco</th>
                  </tr>
                </thead>
                <tbody>
                  {Tablevisible && placeInfoList.map(place => (
                    <tr className="text-center" key={place.id}>
                      <td>{place.name}</td>
                      <td>{place.tag}</td>
                      <td>{place.city}</td>
                      {ecolist.includes(place.id) ? (
                        <td><img src={logo} alt="react-logo" width="30" height="30" /></td>
                      ) : (
                        <td></td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </Table>
              <div className="text-center">
                <p>11h(default) - 2h * 2(일정 개수, 숙소 제외) = 7h</p>
                <h5> 현재 가용시간 : 7h </h5>
              </div>
            </CardBody>
          </Card>
        </Col>
      </CardBody>
      <div className="ml-auto mr-auto" lg="8">
        <Row>
          <Col md="4">
            {/* 이전 페이지로 이동하는 버튼 */}
            <Link to="/admin/icons" className="btn btn-primary btn-round">Prev</Link>
          </Col>
          <Col md="4">
            {/* 공백 */}
          </Col>
          <Col md="4">
            {/* 다음 페이지로 이동하는 버튼 */}
            <Link to="/admin/maps" className="btn btn-primary btn-round">Next</Link>
          </Col>
        </Row>
      </div>
    </Card>
    </Col>
        </Row>

      </div>
  );

export default Plan1;
