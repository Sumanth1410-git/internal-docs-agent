"""
Comprehensive tests for Enterprise Approval Workflow System
Tests approval creation, processing, and enterprise workflow management
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import os
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from uuid import uuid4

# Import test utilities
from tests import TEST_DATA_DIR, create_test_documents, cleanup_test_files

@pytest.fixture
def temp_approval_dir():
    """Create temporary directory for approval system data"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_approval_config():
    """Sample approval configuration for testing"""
    return {
        'approver_roles': {
            'hr': {
                'users': ['HR_MANAGER_123', 'HR_DIRECTOR_456'],
                'required_approvals': 1,
                'escalation_hours': 24
            },
            'technical': {
                'users': ['TECH_LEAD_789', 'CTO_012'],
                'required_approvals': 1,
                'escalation_hours': 12
            },
            'policy': {
                'users': ['POLICY_MANAGER_345', 'LEGAL_TEAM_678'],
                'required_approvals': 2,
                'escalation_hours': 48
            }
        }
    }

class TestEnterpriseApprovalWorkflow:
    """Test suite for Enterprise Approval Workflow System"""
    
    def test_approval_system_initialization(self, temp_approval_dir):
        """Test approval system initialization"""
        from approval_system import EnterpriseApprovalWorkflow
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        assert workflow.data_dir == temp_approval_dir
        assert workflow.data_dir.exists()
        assert hasattr(workflow, 'config')
        assert hasattr(workflow, 'active_requests')
        assert hasattr(workflow, 'approval_history')
    
    def test_approval_configuration_loading(self, temp_approval_dir, sample_approval_config):
        """Test loading of approval configuration"""
        from approval_system import EnterpriseApprovalWorkflow
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Verify default configuration structure
        assert 'approver_roles' in workflow.config
        assert 'approval_rules' in workflow.config
        
        # Check specific role configurations
        assert 'hr' in workflow.config['approver_roles']
        assert 'technical' in workflow.config['approver_roles']
        assert 'general' in workflow.config['approver_roles']
    
    def test_create_approval_request_document_upload(self, temp_approval_dir):
        """Test creating approval request for document upload"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={
                'file_name': 'sensitive_document.pdf',
                'file_size_mb': 8.5,
                'file_type': '.pdf',
                'preview': 'This document contains confidential information'
            },
            category='policy'
        )
        
        assert request_id is not None
        assert request_id in workflow.active_requests
        
        request = workflow.active_requests[request_id]
        assert request['requester_id'] == 'U12345'
        assert request['approval_type'] == ApprovalType.DOCUMENT_UPLOAD.value
        assert request['category'] == 'policy'
        assert request['status'] == 'pending'
    
    def test_auto_approval_for_small_files(self, temp_approval_dir):
        """Test auto-approval mechanism for small, safe files"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Small text file should be auto-approved
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={
                'file_name': 'small_note.txt',
                'file_size_mb': 0.1,
                'file_type': '.txt',
                'preview': 'Just a small note about meeting times'
            },
            category='general'
        )
        
        # Should be auto-approved
        assert request_id.startswith('auto_')
        
        # Check approval history
        auto_approval = next(
            (req for req in workflow.approval_history if req['id'] == request_id),
            None
        )
        assert auto_approval is not None
        assert auto_approval['status'] == 'approved'
        assert auto_approval['approval_method'] == 'automatic'
    
    def test_approval_blocks_generation(self, temp_approval_dir):
        """Test generation of Slack approval blocks"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Create request that requires approval
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={
                'file_name': 'large_document.pdf',
                'file_size_mb': 15.0,
                'file_type': '.pdf'
            },
            category='technical'
        )
        
        blocks = workflow.get_approval_blocks(request_id)
        
        assert len(blocks) > 0
        assert any(block['type'] == 'header' for block in blocks)
        assert any(block['type'] == 'actions' for block in blocks)
        
        # Check for approval buttons
        action_block = next(block for block in blocks if block['type'] == 'actions')
        button_texts = [elem['text']['text'] for elem in action_block['elements']]
        assert 'Approve' in str(button_texts)
        assert 'Reject' in str(button_texts)
    
    def test_approval_response_processing(self, temp_approval_dir):
        """Test processing of approval responses"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Create request
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'test.pdf', 'file_size_mb': 5.0},
            category='hr'
        )
        
        # Get approver from configuration
        request = workflow.active_requests[request_id]
        approver_id = request['approvers'][0]
        
        # Process approval
        result = workflow.process_approval_response(
            request_id=request_id,
            approver_id=approver_id,
            action='approve',
            comments='Looks good to proceed'
        )
        
        assert result['success'] is True
        assert result['status'] == 'approved'
        
        # Request should be moved to history
        assert request_id not in workflow.active_requests
        assert any(req['id'] == request_id for req in workflow.approval_history)
    
    def test_approval_rejection_handling(self, temp_approval_dir):
        """Test handling of approval rejections"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Create request
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'test.pdf'},
            category='policy'
        )
        
        request = workflow.active_requests[request_id]
        approver_id = request['approvers'][0]
        
        # Reject the request
        result = workflow.process_approval_response(
            request_id=request_id,
            approver_id=approver_id,
            action='reject',
            comments='Document contains sensitive information'
        )
        
        assert result['success'] is True
        assert result['status'] == 'rejected'
        
        # Check rejection was recorded
        history_entry = next(req for req in workflow.approval_history if req['id'] == request_id)
        assert history_entry['status'] == 'rejected'
    
    def test_multi_approver_requirement(self, temp_approval_dir):
        """Test requests requiring multiple approvers"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Policy category requires 2 approvals
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.POLICY_CHANGE,
            content={'policy_name': 'Remote Work Policy Update'},
            category='policy'
        )
        
        request = workflow.active_requests[request_id]
        approver1 = request['approvers'][0]
        approver2 = request['approvers'][1]
        
        # First approval
        result1 = workflow.process_approval_response(
            request_id=request_id,
            approver_id=approver1,
            action='approve'
        )
        
        assert result1['success'] is True
        assert result1['status'] == 'pending'  # Still needs more approvals
        assert result1['approvals_needed'] == 1
        
        # Second approval
        result2 = workflow.process_approval_response(
            request_id=request_id,
            approver_id=approver2,
            action='approve'
        )
        
        assert result2['success'] is True
        assert result2['status'] == 'approved'  # Now fully approved
    
    def test_unauthorized_approver_handling(self, temp_approval_dir):
        """Test handling of unauthorized approval attempts"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'test.pdf'},
            category='hr'
        )
        
        # Try to approve with unauthorized user
        result = workflow.process_approval_response(
            request_id=request_id,
            approver_id='UNAUTHORIZED_USER',
            action='approve'
        )
        
        assert result['success'] is False
        assert 'not authorized' in result['message'].lower()
    
    def test_expired_request_handling(self, temp_approval_dir):
        """Test handling of expired approval requests"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'test.pdf'},
            category='hr'
        )
        
        # Manually set expiration to past
        request = workflow.active_requests[request_id]
        past_time = datetime.now() - timedelta(hours=1)
        request['expires_at'] = past_time.isoformat()
        
        approver_id = request['approvers'][0]
        
        # Try to approve expired request
        result = workflow.process_approval_response(
            request_id=request_id,
            approver_id=approver_id,
            action='approve'
        )
        
        assert result['success'] is False
        assert 'expired' in result['message'].lower()
    
    def test_request_escalation(self, temp_approval_dir):
        """Test automatic escalation of expired requests"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'test.pdf'},
            category='technical'
        )
        
        # Manually expire the request
        request = workflow.active_requests[request_id]
        past_time = datetime.now() - timedelta(hours=1)
        request['expires_at'] = past_time.isoformat()
        
        # Check expired requests
        expired_count = workflow.check_expired_requests()
        
        assert expired_count == 1
        
        # Verify escalation
        escalated_request = workflow.active_requests[request_id]
        assert escalated_request['status'] == 'escalated'
        assert escalated_request['escalation_level'] == 1
    
    def test_pending_approvals_for_user(self, temp_approval_dir):
        """Test retrieval of pending approvals for specific user"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Create multiple requests with different approvers
        request1 = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'test1.pdf'},
            category='hr'
        )
        
        request2 = workflow.create_approval_request(
            requester_id='U67890',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'test2.pdf'},
            category='technical'
        )
        
        # Get pending approvals for HR manager
        hr_approver = workflow.config['approver_roles']['hr']['users'][0]
        pending_hr = workflow.get_pending_approvals_for_user(hr_approver)
        
        assert len(pending_hr) >= 1
        assert any(req['id'] == request1 for req in pending_hr)
    
    def test_approval_statistics(self, temp_approval_dir):
        """Test approval system statistics generation"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Create and process several requests
        for i in range(3):
            request_id = workflow.create_approval_request(
                requester_id=f'U{i}',
                approval_type=ApprovalType.DOCUMENT_UPLOAD,
                content={'file_name': f'test{i}.pdf'},
                category='hr'
            )
            
            # Approve one, reject one, leave one pending
            if i == 0:
                request = workflow.active_requests[request_id]
                approver_id = request['approvers'][0]
                workflow.process_approval_response(request_id, approver_id, 'approve')
            elif i == 1:
                request = workflow.active_requests[request_id]
                approver_id = request['approvers'][0]
                workflow.process_approval_response(request_id, approver_id, 'reject')
        
        stats = workflow.get_approval_statistics()
        
        assert 'total_processed' in stats
        assert 'approval_rate' in stats
        assert 'rejection_rate' in stats
        assert 'pending_count' in stats
        assert stats['total_processed'] == 2  # 2 processed, 1 pending
        assert stats['pending_count'] == 1
    
    def test_sensitive_query_approval(self, temp_approval_dir):
        """Test approval workflow for sensitive queries"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Sensitive query should require approval
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.SENSITIVE_QUERY,
            content={
                'query': 'What are the salary ranges for different positions?',
                'sensitivity_level': 'high'
            },
            category='hr'
        )
        
        assert request_id is not None
        assert not request_id.startswith('auto_')  # Should not be auto-approved
        
        request = workflow.active_requests[request_id]
        assert request['approval_type'] == ApprovalType.SENSITIVE_QUERY.value
        assert request['category'] == 'hr'
    
    def test_bulk_operation_approval(self, temp_approval_dir):
        """Test approval workflow for bulk operations"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.BULK_OPERATION,
            content={
                'operation_type': 'mass_document_update',
                'affected_documents': 50,
                'description': 'Update all policy documents with new compliance requirements'
            },
            category='policy'
        )
        
        assert request_id is not None
        request = workflow.active_requests[request_id]
        assert request['approval_type'] == ApprovalType.BULK_OPERATION.value
        assert request['required_approvals'] == 2  # Policy category requires 2 approvals
    
    def test_data_persistence(self, temp_approval_dir):
        """Test data persistence across workflow instances"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        # Create first workflow instance
        workflow1 = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        request_id = workflow1.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'persistence_test.pdf'},
            category='hr'
        )
        
        # Create second workflow instance (simulating restart)
        workflow2 = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Should load existing requests
        assert request_id in workflow2.active_requests
        assert workflow2.active_requests[request_id]['requester_id'] == 'U12345'
    
    def test_approval_request_validation(self, temp_approval_dir):
        """Test validation of approval request parameters"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Test with invalid approval type
        try:
            workflow.create_approval_request(
                requester_id='U12345',
                approval_type='invalid_type',  # Invalid type
                content={'test': 'data'},
                category='general'
            )
        except (ValueError, AttributeError):
            pass  # Expected to fail
        
        # Test with empty requester ID
        request_id = workflow.create_approval_request(
            requester_id='',  # Empty requester
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'test.pdf'},
            category='general'
        )
        
        # Should still create request but with validation warnings
        assert request_id is not None

class TestApprovalWorkflowIntegration:
    """Integration tests for approval workflow with other components"""
    
    def test_integration_with_file_processor(self, temp_approval_dir):
        """Test integration between approval workflow and file processor"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Simulate file processor requesting approval
        file_metadata = {
            'file_name': 'confidential_report.pdf',
            'file_size_mb': 12.5,
            'file_type': '.pdf',
            'uploaded_by': 'U12345',
            'contains_sensitive_keywords': True
        }
        
        request_id = workflow.create_approval_request(
            requester_id=file_metadata['uploaded_by'],
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content=file_metadata,
            category='policy'
        )
        
        assert request_id is not None
        request = workflow.active_requests[request_id]
        assert request['content']['file_name'] == 'confidential_report.pdf'
    
    def test_integration_with_slack_bot(self, temp_approval_dir):
        """Test integration between approval workflow and Slack bot"""
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Create request
        request_id = workflow.create_approval_request(
            requester_id='U12345',
            approval_type=ApprovalType.DOCUMENT_UPLOAD,
            content={'file_name': 'slack_integration_test.pdf'},
            category='general'
        )
        
        # Generate Slack blocks (as would be done by Slack bot)
        blocks = workflow.get_approval_blocks(request_id)
        
        # Verify blocks are suitable for Slack
        assert isinstance(blocks, list)
        assert len(blocks) > 0
        
        # Check block structure
        for block in blocks:
            assert 'type' in block
            assert block['type'] in ['header', 'section', 'actions', 'context']
    
    @pytest.mark.asyncio
    async def test_concurrent_approval_processing(self, temp_approval_dir):
        """Test concurrent approval request processing"""
        import asyncio
        from approval_system import EnterpriseApprovalWorkflow, ApprovalType
        
        workflow = EnterpriseApprovalWorkflow(str(temp_approval_dir))
        
        # Create multiple approval requests concurrently
        async def create_request_task(user_id):
            return workflow.create_approval_request(
                requester_id=f'U{user_id}',
                approval_type=ApprovalType.DOCUMENT_UPLOAD,
                content={'file_name': f'concurrent_test_{user_id}.pdf'},
                category='hr'
            )
        
        # Process multiple requests
        tasks = [create_request_task(i) for i in range(5)]
        request_ids = await asyncio.gather(*tasks)
        
        # Verify all requests were created
        assert len(request_ids) == 5
        assert all(req_id in workflow.active_requests for req_id in request_ids)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
