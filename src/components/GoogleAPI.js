import React from "react";
import GoogleLogin from "react-google-login";


const clientId = "201040501851-rb5dcr909h6mla2a34on9u2rgo0a7fa1.apps.googleusercontent.com";

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
                scope='https://www.googleapis.com/auth/drive.photos.readonly'
                buttonText='Login'
                onSuccess={responseGoogle}
                onFailure={responseGoogle}
                cookiePolicy={"single_host_origin"}
            />
        );
    }
}

export default GoogleAPI;
