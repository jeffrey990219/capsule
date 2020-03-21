import React from "react";
import GoogleLogin from "react-google-login";


const clientId = "839718518795-advg62g6djcnggvn6dm66idufa1pa8nk.apps.googleusercontent.com";

const responseGoogle = response => {
    console.log(response);
    console.log(response.tokenObj.access_token);

    fetch("https://photoslibrary.googleapis.com/v1/albums/")
        .then(response => response.json())
        .then(json => {
            console.log(json.tokenObj.access_token);
        });
};

class GoogleAPI extends React.Component {
    constructor(props) {
        super(props);
        this.state = { LoggedIn: false };
    }
    render() {
        return (
            <GoogleLogin
                clientId={clientId}
                scope='https://www.googleapis.com/auth/photoslibrary.readonly'
                buttonText='Login'
                onSuccess={responseGoogle}
                onFailure={responseGoogle}
                cookiePolicy={"single_host_origin"}
            />
        );
    }
}

export default GoogleAPI;
