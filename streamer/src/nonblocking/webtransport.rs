//! WebTransport over HTTP/3 implementation for browser-based transaction submission.
//!
//! WebTransport uses ALPN "h3" and requires HTTP/3 SETTINGS exchange followed by
//! a CONNECT request before application streams can be opened.

use {
    bytes::{Buf, BufMut, Bytes, BytesMut},
    quinn::{Connection, RecvStream, SendStream},
};

pub const ALPN_WEBTRANSPORT: &[u8] = b"h3";

const STREAM_TYPE_CONTROL: u64 = 0x00;
const STREAM_TYPE_WEBTRANSPORT: u64 = 0x54;

const FRAME_HEADERS: u64 = 0x01;
const FRAME_SETTINGS: u64 = 0x04;
const FRAME_WEBTRANSPORT: u64 = 0x41;

const SETTING_ENABLE_CONNECT_PROTOCOL: u64 = 0x08;
const SETTING_WEBTRANSPORT_MAX_SESSIONS: u64 = 0xc671706a;

#[derive(Debug, thiserror::Error)]
pub enum WebTransportError {
    #[error("Connection error: {0}")]
    Connection(#[from] quinn::ConnectionError),
    #[error("Read error: {0}")]
    Read(#[from] quinn::ReadError),
    #[error("Write error: {0}")]
    Write(#[from] quinn::WriteError),
    #[error("Unexpected end of stream")]
    UnexpectedEnd,
    #[error("Unexpected stream type: {0}")]
    UnexpectedStreamType(u64),
    #[error("Unexpected frame type: {0}")]
    UnexpectedFrame(u64),
    #[error("WebTransport not supported by peer")]
    NotSupported,
}

fn decode_varint<B: Buf>(buf: &mut B) -> Option<u64> {
    if !buf.has_remaining() {
        return None;
    }
    let first = buf.get_u8();
    let len = 1 << (first >> 6);
    if buf.remaining() + 1 < len {
        return None;
    }
    let mut val = u64::from(first & 0x3f);
    for _ in 1..len {
        val = (val << 8) | u64::from(buf.get_u8());
    }
    Some(val)
}

fn encode_varint<B: BufMut>(buf: &mut B, val: u64) {
    if val < 0x40 {
        buf.put_u8(val as u8);
    } else if val < 0x4000 {
        buf.put_u16(0x4000 | val as u16);
    } else if val < 0x40000000 {
        buf.put_u32(0x80000000 | val as u32);
    } else {
        buf.put_u64(0xc000000000000000 | val);
    }
}

/// Computes the WebTransport session ID from a QUIC stream ID.
fn compute_session_id(stream_id: quinn::StreamId) -> u64 {
    stream_id.index() * 4 + u64::from(!stream_id.initiator().is_client())
}

pub struct WebTransportSession {
    conn: Connection,
    session_id: u64,
    // HTTP/3 control streams must remain open for the session lifetime.
    // Dropping these would close the streams and terminate the session.
    _control_send: SendStream,
    _control_recv: RecvStream,
    // The CONNECT request/response streams must also stay open.
    _connect_send: SendStream,
    _connect_recv: RecvStream,
}

impl WebTransportSession {
    /// Accepts a WebTransport session on an established QUIC connection [server-side].
    pub async fn accept(conn: Connection) -> Result<Self, WebTransportError> {
        let (control_send, control_recv) = tokio::try_join!(
            Self::send_server_settings(&conn),
            Self::recv_settings(&conn)
        )?;

        let (connect_send, connect_recv, session_id) = Self::accept_connect(&conn).await?;

        Ok(Self {
            conn,
            session_id,
            _control_send: control_send,
            _control_recv: control_recv,
            _connect_send: connect_send,
            _connect_recv: connect_recv,
        })
    }

    /// Connects to a WebTransport server [client-side].
    pub async fn connect(conn: Connection, path: &str) -> Result<Self, WebTransportError> {
        let (control_send, control_recv) = tokio::try_join!(
            Self::send_client_settings(&conn),
            Self::recv_settings(&conn)
        )?;

        let (connect_send, connect_recv, session_id) = Self::send_connect(&conn, path).await?;

        Ok(Self {
            conn,
            session_id,
            _control_send: control_send,
            _control_recv: control_recv,
            _connect_send: connect_send,
            _connect_recv: connect_recv,
        })
    }

    async fn accept_connect(
        conn: &Connection,
    ) -> Result<(SendStream, RecvStream, u64), WebTransportError> {
        let (mut send, mut recv) = conn.accept_bi().await?;
        let session_id = compute_session_id(recv.id());

        let mut buf = BytesMut::with_capacity(1024);

        loop {
            let chunk = recv
                .read_chunk(1024, true)
                .await?
                .ok_or(WebTransportError::UnexpectedEnd)?;
            buf.extend_from_slice(&chunk.bytes);

            let mut slice = buf.as_ref();
            let Some(frame_type) = decode_varint(&mut slice) else {
                continue;
            };
            if frame_type != FRAME_HEADERS {
                return Err(WebTransportError::UnexpectedFrame(frame_type));
            }

            let Some(frame_len) = decode_varint(&mut slice) else {
                continue;
            };

            if slice.len() >= frame_len as usize {
                break;
            }
        }

        send.write_all(&build_200_response()).await?;
        Ok((send, recv, session_id))
    }

    async fn send_connect(
        conn: &Connection,
        path: &str,
    ) -> Result<(SendStream, RecvStream, u64), WebTransportError> {
        let (mut send, mut recv) = conn.open_bi().await?;
        let session_id = compute_session_id(send.id());

        send.write_all(&build_connect_request(path)).await?;

        let mut buf = BytesMut::with_capacity(256);

        loop {
            let chunk = recv
                .read_chunk(256, true)
                .await?
                .ok_or(WebTransportError::UnexpectedEnd)?;
            buf.extend_from_slice(&chunk.bytes);

            let mut slice = buf.as_ref();
            let Some(frame_type) = decode_varint(&mut slice) else {
                continue;
            };
            if frame_type != FRAME_HEADERS {
                return Err(WebTransportError::UnexpectedFrame(frame_type));
            }

            let Some(frame_len) = decode_varint(&mut slice) else {
                continue;
            };

            if slice.len() >= frame_len as usize {
                break;
            }
        }

        Ok((send, recv, session_id))
    }

    async fn send_server_settings(conn: &Connection) -> Result<SendStream, WebTransportError> {
        let mut send = conn.open_uni().await?;

        let mut buf = BytesMut::with_capacity(32);
        encode_varint(&mut buf, STREAM_TYPE_CONTROL);
        encode_varint(&mut buf, FRAME_SETTINGS);

        let mut payload = BytesMut::with_capacity(16);
        encode_varint(&mut payload, SETTING_ENABLE_CONNECT_PROTOCOL);
        encode_varint(&mut payload, 1);
        encode_varint(&mut payload, SETTING_WEBTRANSPORT_MAX_SESSIONS);
        encode_varint(&mut payload, 1);

        encode_varint(&mut buf, payload.len() as u64);
        buf.extend_from_slice(&payload);

        send.write_all(&buf).await?;
        Ok(send)
    }

    async fn send_client_settings(conn: &Connection) -> Result<SendStream, WebTransportError> {
        let mut send = conn.open_uni().await?;

        let mut buf = BytesMut::with_capacity(32);
        encode_varint(&mut buf, STREAM_TYPE_CONTROL);
        encode_varint(&mut buf, FRAME_SETTINGS);

        let mut payload = BytesMut::with_capacity(16);
        encode_varint(&mut payload, SETTING_WEBTRANSPORT_MAX_SESSIONS);
        encode_varint(&mut payload, 1);

        encode_varint(&mut buf, payload.len() as u64);
        buf.extend_from_slice(&payload);

        send.write_all(&buf).await?;
        Ok(send)
    }

    async fn recv_settings(conn: &Connection) -> Result<RecvStream, WebTransportError> {
        let mut recv = conn.accept_uni().await?;
        let mut buf = BytesMut::with_capacity(256);

        // Read until we have the stream type
        loop {
            let chunk = recv
                .read_chunk(256, true)
                .await?
                .ok_or(WebTransportError::UnexpectedEnd)?;
            buf.extend_from_slice(&chunk.bytes);

            let mut slice = buf.as_ref();
            if let Some(stream_type) = decode_varint(&mut slice) {
                if stream_type != STREAM_TYPE_CONTROL {
                    return Err(WebTransportError::UnexpectedStreamType(stream_type));
                }
                break;
            }
        }

        // Read SETTINGS frame
        loop {
            let mut slice = buf.as_ref();
            decode_varint(&mut slice); // skip stream type

            if slice.remaining() < 2 {
                let chunk = recv
                    .read_chunk(256, true)
                    .await?
                    .ok_or(WebTransportError::UnexpectedEnd)?;
                buf.extend_from_slice(&chunk.bytes);
                continue;
            }

            let frame_type = decode_varint(&mut slice).ok_or(WebTransportError::UnexpectedEnd)?;
            if frame_type != FRAME_SETTINGS {
                return Err(WebTransportError::UnexpectedFrame(frame_type));
            }

            let frame_len = decode_varint(&mut slice).ok_or(WebTransportError::UnexpectedEnd)?;

            // Read more if needed
            while slice.len() < frame_len as usize {
                let chunk = recv
                    .read_chunk(256, true)
                    .await?
                    .ok_or(WebTransportError::UnexpectedEnd)?;
                buf.extend_from_slice(&chunk.bytes);
                // Re-parse after extending buffer
                slice = buf.as_ref();
                decode_varint(&mut slice); // stream type
                decode_varint(&mut slice); // frame type
                decode_varint(&mut slice); // frame len
            }

            // Parse settings
            let mut webtransport_supported = false;
            let settings_end = frame_len as usize;
            let mut consumed = 0;

            while consumed < settings_end {
                let setting_id =
                    decode_varint(&mut slice).ok_or(WebTransportError::UnexpectedEnd)?;
                let setting_val =
                    decode_varint(&mut slice).ok_or(WebTransportError::UnexpectedEnd)?;
                consumed = settings_end - slice.len();

                if setting_id == SETTING_WEBTRANSPORT_MAX_SESSIONS && setting_val > 0 {
                    webtransport_supported = true;
                }
            }

            if !webtransport_supported {
                return Err(WebTransportError::NotSupported);
            }

            break;
        }

        Ok(recv)
    }

    /// Accept a unidirectional WebTransport stream.
    pub async fn accept_uni(&self) -> Result<RecvStream, WebTransportError> {
        'outer: loop {
            let mut recv = self.conn.accept_uni().await?;

            // Read header byte-by-byte to avoid consuming application data.
            // Header format: stream_type (varint) + session_id (varint), typically 2-4 bytes.
            let mut header = BytesMut::with_capacity(16);

            // Read first byte to determine stream type varint length
            let mut byte = [0u8; 1];
            if recv.read_exact(&mut byte).await.is_err() {
                continue 'outer;
            }
            header.extend_from_slice(&byte);

            let type_len = 1 << (byte[0] >> 6);
            for _ in 1..type_len {
                if recv.read_exact(&mut byte).await.is_err() {
                    continue 'outer;
                }
                header.extend_from_slice(&byte);
            }

            let mut slice = header.as_ref();
            let Some(stream_type) = decode_varint(&mut slice) else {
                continue 'outer;
            };

            match stream_type {
                STREAM_TYPE_WEBTRANSPORT => {
                    // Read session ID varint
                    if recv.read_exact(&mut byte).await.is_err() {
                        continue 'outer;
                    }
                    header.extend_from_slice(&byte);
                    let sid_len = 1 << (byte[0] >> 6);
                    for _ in 1..sid_len {
                        if recv.read_exact(&mut byte).await.is_err() {
                            continue 'outer;
                        }
                        header.extend_from_slice(&byte);
                    }

                    let mut sid_slice = &header[type_len..];
                    let Some(sid) = decode_varint(&mut sid_slice) else {
                        continue 'outer;
                    };

                    if sid != self.session_id {
                        continue 'outer;
                    }
                    return Ok(recv);
                }
                STREAM_TYPE_CONTROL | 0x02 | 0x03 => continue 'outer,
                _ => continue 'outer,
            }
        }
    }

    /// Accept a bidirectional WebTransport stream.
    pub async fn accept_bi(&self) -> Result<(SendStream, RecvStream), WebTransportError> {
        loop {
            let (send, mut recv) = self.conn.accept_bi().await?;
            let mut buf = BytesMut::with_capacity(16);

            loop {
                let chunk = recv
                    .read_chunk(16, true)
                    .await?
                    .ok_or(WebTransportError::UnexpectedEnd)?;
                buf.extend_from_slice(&chunk.bytes);

                let mut slice = buf.as_ref();
                let Some(frame_type) = decode_varint(&mut slice) else {
                    continue;
                };

                if frame_type != FRAME_WEBTRANSPORT {
                    break;
                }

                let Some(sid) = decode_varint(&mut slice) else {
                    continue;
                };
                if sid != self.session_id {
                    break;
                }

                return Ok((send, recv));
            }
        }
    }

    /// Open a unidirectional stream.
    pub async fn open_uni(&self) -> Result<SendStream, WebTransportError> {
        let mut send = self.conn.open_uni().await?;

        let mut header = BytesMut::with_capacity(16);
        encode_varint(&mut header, STREAM_TYPE_WEBTRANSPORT);
        encode_varint(&mut header, self.session_id);
        send.write_all(&header).await?;

        Ok(send)
    }

    /// Open a bidirectional stream.
    pub async fn open_bi(&self) -> Result<(SendStream, RecvStream), WebTransportError> {
        let (mut send, recv) = self.conn.open_bi().await?;

        let mut header = BytesMut::with_capacity(16);
        encode_varint(&mut header, FRAME_WEBTRANSPORT);
        encode_varint(&mut header, self.session_id);
        send.write_all(&header).await?;

        Ok((send, recv))
    }

    pub fn close(&self, code: u32, reason: &[u8]) {
        self.conn.close(quinn::VarInt::from_u32(code), reason);
    }

    pub fn connection(&self) -> &Connection {
        &self.conn
    }

    pub fn remote_address(&self) -> std::net::SocketAddr {
        self.conn.remote_address()
    }

    pub fn session_id(&self) -> u64 {
        self.session_id
    }
}

fn build_200_response() -> Bytes {
    let mut buf = BytesMut::with_capacity(32);
    encode_varint(&mut buf, FRAME_HEADERS);

    // QPACK-encoded ":status: 200" (static table index 25)
    let qpack_headers: &[u8] = &[0x00, 0x00, 0xd9];

    encode_varint(&mut buf, qpack_headers.len() as u64);
    buf.extend_from_slice(qpack_headers);
    buf.freeze()
}

fn build_connect_request(path: &str) -> Bytes {
    let mut buf = BytesMut::with_capacity(128);
    encode_varint(&mut buf, FRAME_HEADERS);

    let mut qpack = BytesMut::with_capacity(64);

    // Required Insert Count = 0, Delta Base = 0
    qpack.put_u8(0x00);
    qpack.put_u8(0x00);

    // :method = CONNECT (static index 15)
    qpack.put_u8(0xcf);
    // :scheme = https (static index 23)
    qpack.put_u8(0xd7);

    // :protocol = webtransport (literal)
    qpack.put_u8(0x27);
    qpack.put_u8(0x09);
    qpack.extend_from_slice(b":protocol");
    qpack.put_u8(0x0c);
    qpack.extend_from_slice(b"webtransport");

    // :authority = localhost (static index 0)
    qpack.put_u8(0x50);
    qpack.put_u8(0x09);
    qpack.extend_from_slice(b"localhost");

    // :path (static index 1)
    qpack.put_u8(0x51);
    let path_bytes = path.as_bytes();
    qpack.put_u8(path_bytes.len() as u8);
    qpack.extend_from_slice(path_bytes);

    encode_varint(&mut buf, qpack.len() as u64);
    buf.extend_from_slice(&qpack);
    buf.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encoding_single_byte() {
        let mut buf = BytesMut::new();
        encode_varint(&mut buf, 0);
        assert_eq!(buf.as_ref(), &[0x00]);

        buf.clear();
        encode_varint(&mut buf, 63);
        assert_eq!(buf.as_ref(), &[0x3f]);
    }

    #[test]
    fn test_varint_encoding_two_bytes() {
        let mut buf = BytesMut::new();
        encode_varint(&mut buf, 64);
        assert_eq!(buf.as_ref(), &[0x40, 0x40]);

        buf.clear();
        encode_varint(&mut buf, 16383);
        assert_eq!(buf.as_ref(), &[0x7f, 0xff]);
    }

    #[test]
    fn test_varint_encoding_four_bytes() {
        let mut buf = BytesMut::new();
        encode_varint(&mut buf, 16384);
        assert_eq!(buf.as_ref(), &[0x80, 0x00, 0x40, 0x00]);
    }

    #[test]
    fn test_varint_decoding_single_byte() {
        let data = [0x25u8];
        let mut slice = data.as_ref();
        assert_eq!(decode_varint(&mut slice), Some(37));
    }

    #[test]
    fn test_varint_decoding_two_bytes() {
        let data = [0x7b, 0xbdu8];
        let mut slice = data.as_ref();
        assert_eq!(decode_varint(&mut slice), Some(15293));
    }

    #[test]
    fn test_varint_roundtrip() {
        for val in [0, 1, 63, 64, 16383, 16384, 1073741823, 1073741824] {
            let mut buf = BytesMut::new();
            encode_varint(&mut buf, val);
            let mut slice = buf.as_ref();
            assert_eq!(decode_varint(&mut slice), Some(val));
        }
    }

    #[test]
    fn test_build_200_response() {
        let response = build_200_response();
        // Should start with HEADERS frame type (0x01)
        assert_eq!(response[0], 0x01);
        // Response should be non-empty
        assert!(response.len() > 3);
    }

    #[test]
    fn test_build_connect_request() {
        let request = build_connect_request("/");
        // Should start with HEADERS frame type (0x01)
        assert_eq!(request[0], 0x01);
        // Request should contain the path
        assert!(request.len() > 10);
    }

    #[test]
    fn test_build_connect_request_with_path() {
        let request = build_connect_request("/submit");
        // Should start with HEADERS frame type (0x01)
        assert_eq!(request[0], 0x01);
        // Request should be larger with longer path
        let short_request = build_connect_request("/");
        assert!(request.len() > short_request.len());
    }
}
