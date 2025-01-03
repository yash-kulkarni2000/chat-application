import React from "react";
import './Modal.css';

const Modal = ({ isOpen, onClose, onConfirm }) => {
    if (!isOpen) return null;

    return (
        <div className="modal-overlay">
            <div className="modal">
                <h2>Warning</h2>
                <p>Your message contains potentially hateful content. Are you sure want to send it?</p>
                <div className="modal-actions">
                    <button onClick={onConfirm} className="confirm-btn">Yes</button>
                    <button onClick={onClose} className="cancel-btn">No</button>
                </div>
            </div>
        </div>
    );
};

export default Modal;