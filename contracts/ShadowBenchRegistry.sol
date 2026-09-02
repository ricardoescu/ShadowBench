// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract ShadowBenchRegistry {

    struct AuditCommitment {
        bytes32 manifestHash;
        bytes32 resultsRoot;
        address auditor;
        uint256 timestamp;
    }

    mapping(bytes32 => AuditCommitment) public audits;

    event AuditCommitted(
        bytes32 indexed auditId,
        bytes32 manifestHash,
        bytes32 resultsRoot,
        address indexed auditor,
        uint256 timestamp
    );

    function commitAudit(
        bytes32 auditId,
        bytes32 manifestHash,
        bytes32 resultsRoot
    ) external {
        require(
            audits[auditId].timestamp == 0,
            "Audit already committed"
        );

        audits[auditId] = AuditCommitment({
            manifestHash: manifestHash,
            resultsRoot: resultsRoot,
            auditor: msg.sender,
            timestamp: block.timestamp
        });

        emit AuditCommitted(
            auditId,
            manifestHash,
            resultsRoot,
            msg.sender,
            block.timestamp
        );
    }
}
